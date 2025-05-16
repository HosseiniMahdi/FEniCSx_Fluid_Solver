import os
import ufl
import csv
import gmsh
import time 
import psutil
import pyvista 
import dolfinx
import numpy as np
import matplotlib.pyplot as plt

from dolfinx import la
from mpi4py import MPI
from pathlib import Path
from petsc4py import PETSc
from basix.ufl import element
from dolfinx.mesh import Mesh
from dolfinx.plot import vtk_mesh
from dolfinx.io import (VTXWriter, gmshio,VTKFile)
from dolfinx.fem.petsc import (assemble_matrix_block, assemble_vector_block)
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
from ufl import (FacetNormal, Measure, TestFunction, TrialFunction, as_vector, div, dx, ds, inner, grad)
from dolfinx.fem import (Constant, Function, functionspace, assemble_scalar, dirichletbc, form, locate_dofs_topological, set_bc)





# Initialize gmsh and create geometry 
gmsh.initialize()
gmsh.option.setNumber("General.Verbosity", 0)

u_deg = 2  # velocity element polynomial degree
p_deg = 1  # pressure element polynomial degree



L = 2.2 
H = 0.41
c_x = c_y = 0.2
r = 0.05
gdim = 2

mesh_comm = MPI.COMM_WORLD
model_rank = 0 
if mesh_comm.rank == model_rank:
    rectangle = gmsh.model.occ.addRectangle(0, 0, 0, L, H, tag=1)   
    obstacle = gmsh.model.occ.addDisk(c_x, c_y, 0, r, r)

if mesh_comm.rank == model_rank:
    fluid = gmsh.model.occ.cut([(gdim, rectangle)], [(gdim, obstacle)]) # cut the cylinder from the rectangle
    gmsh.model.occ.synchronize()
    
fluid_marker = 1 # A Marker for the fluid so it can be recalled
if mesh_comm.rank == model_rank:
    volumes = gmsh.model.getEntities(dim=gdim)
    assert (len(volumes) == 1)   # Check that we have one entery for "volumes" if we had two, it meant we the cut didt work
    gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], fluid_marker) # Add a physical group to the fluid volume this is needed for the meshing (volumes[0][0] takes the geometrical dimention, volumes[0][1] is the tag of the surface, fluid_marker is the physical group ID)
    gmsh.model.setPhysicalName(volumes[0][0], fluid_marker, "Fluid") # give this physical group a name of "Fluid"

inlet_marker, outlet_marker, wall_marker, obstacle_marker = 2, 3, 4, 5
inflow, outflow, walls, obstacle = [], [], [], []
if mesh_comm.rank == model_rank:
    boundaries = gmsh.model.getBoundary(volumes, oriented=False)
    for boundary in boundaries: #loop goes through the boundaries and checks if the center of mass values are close to the given values and appends the boundary to the corresponding list
        center_of_mass = gmsh.model.occ.getCenterOfMass(boundary[0], boundary[1])
        if np.allclose(center_of_mass, [0, H / 2, 0]):
            inflow.append(boundary[1])
        elif np.allclose(center_of_mass, [L, H / 2, 0]):
            outflow.append(boundary[1])
        elif np.allclose(center_of_mass, [L / 2, H, 0]) or np.allclose(center_of_mass, [L / 2, 0, 0]):
            walls.append(boundary[1])
        else:
            obstacle.append(boundary[1])
                
    gmsh.model.addPhysicalGroup(1, walls, wall_marker)
    gmsh.model.setPhysicalName(1, wall_marker, "Walls")
    gmsh.model.addPhysicalGroup(1, inflow, inlet_marker)
    gmsh.model.setPhysicalName(1, inlet_marker, "Inlet")
    gmsh.model.addPhysicalGroup(1, outflow, outlet_marker)
    gmsh.model.setPhysicalName(1, outlet_marker, "Outlet")
    gmsh.model.addPhysicalGroup(1, obstacle, obstacle_marker)
    gmsh.model.setPhysicalName(1, obstacle_marker, "Obstacle")             




# Get directory where script is located
script_dir = Path(__file__).parent.resolve()

# Add descriptor to indicate higher-order geometry
geometry_tag = "linear_geom"

# Build output folder name
base_output = script_dir / f"DFG_2D-1_results_{geometry_tag}_u{u_deg}_p{p_deg}"
base_output.mkdir(parents=True, exist_ok=True)

# Set log file path
log_filename = base_output / f"simulation_log_{geometry_tag}_u{u_deg}_p{p_deg}.csv"
base_output.mkdir(parents=True, exist_ok=True)

# Path to CSV log file (inside base folder)
log_filename = base_output / f"simulation_log_u{u_deg}_p{p_deg}.csv"
with open(log_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Iteration", "Resolution min", "Resolution max", "Pressure difference", "L_2 Velocity norm",
                         "L_2 Pressure norm", "hmin", "hmax", "V_Dofs", "P_Dofs",
                         "Total DOFs", "Number of Cells", "Drag coefficient", "Lift coefficient", "CPU time", "Memory used"])


# Empty lists for visualizing the results
errors_L2_u = []
errors_L2_p = []
iterations = []
pressure_norm = []
velocity_norm = []
drag_coefficient = []
lift_coefficient = []
min_element_size = []
max_element_size = []
total_number_of_Dofs = []
total_number_of_cells = []
p_diff = np.zeros(35, dtype=PETSc.ScalarType)


for refinement_level in range(35):
    
    # For memory usage
    process = psutil.Process()
    

    print(f"--- Refinement Level: {refinement_level} ---")
    iterations.append(refinement_level)
    
    # start the timer for CPU time
    start_time = time.process_time()
    
    # Memory usage before mesh generation
    mem_before = process.memory_info().rss / (1024 * 1024)  # from bits to MB
    
    
    # we want a variable resolution mesh next to the obsticle. we use Gmsh Fields.
    res_min = r / (2 + refinement_level) 
    res_max = H / (2 + refinement_level) # changed 4 to 2
    
    
    print(f"Minimum defined mesh size: {res_min}")
    print(f"Maximum defined mesh size: {res_max}")
    
    if mesh_comm.rank == model_rank:    
          
        
        distance_field = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(distance_field, "EdgesList", obstacle)
        threshold_field = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(threshold_field, "IField", distance_field)
        gmsh.model.mesh.field.setNumber(threshold_field, "LcMin", res_min) # res_min is the min mesh size at the obstacle
        gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", res_max) # 0.25 is max mesh size at right wall  
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", r) # bellow this element will have size res_min
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 2 * H) # above this element will have size 0.25 * H
        min_field = gmsh.model.mesh.field.add("Min") 
        gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [threshold_field]) 
        gmsh.model.mesh.field.setAsBackgroundMesh(min_field) # sets the thereshhold field for the whole domain
    
        
    # Generating the mesh
    if mesh_comm.rank == model_rank:
        gmsh.model.mesh.generate(gdim)
        gmsh.model.mesh.setOrder(1)
 

    # loading the mesh and boundary markers
    mesh, _, ft = gmshio.model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=gdim)
    ft.name = "Facet markers"






    # function spaces
    v_cg2 = element("Lagrange", mesh.topology.cell_name(), u_deg, shape=(mesh.geometry.dim,))
    v_cg1 = element("Lagrange", mesh.topology.cell_name(), p_deg) 

    V = functionspace(mesh, v_cg2)
    Q = functionspace(mesh, v_cg1)

    fdim = mesh.topology.dim - 1
    

    # Variational formulation
    # Define trial and test functions
    (u, p) = TrialFunction(V), TrialFunction(Q)
    (v, q) = TestFunction(V), TestFunction(Q)
    f = Constant(mesh, (PETSc.ScalarType(0), PETSc.ScalarType(0))) # we give these two scalars because we are using a vector function

    # Define linear and bilinear forms
    a = form([[inner(grad(u), grad(v)) * dx, inner(p, div(v)) * dx], [inner(div(u), q) * dx, None]])
    L = form([inner(f, v) * dx, inner(Constant(mesh, PETSc.ScalarType(0)), q) * dx]) # the second term has zero contribution but its kept anyways


    
    
    # Boundary conditions
    class Inletvelocity():
        def __call__(self, x):
            values = np.zeros((gdim, x.shape[1]), dtype=PETSc.ScalarType)
            values[0] =  1.2 * x[1] * (0.41 - x[1]) / ((0.41 ** 2))
            return values
        
    u_inlet = Function(V)
    inlet_velocity = Inletvelocity()
    u_inlet.interpolate(inlet_velocity)
    bcu_inlet = dirichletbc(u_inlet, locate_dofs_topological(V, fdim, ft.find(inlet_marker)))

    u_nonslip = np.array((0,) * mesh.geometry.dim, dtype=PETSc.ScalarType)
    bcu_wall = dirichletbc(u_nonslip, locate_dofs_topological(V, fdim, ft.find(wall_marker)), V)

    bcu_obstacle = dirichletbc(u_nonslip, locate_dofs_topological(V, fdim, ft.find(obstacle_marker)), V)
    bcu = [bcu_inlet, bcu_wall, bcu_obstacle]

    bcp_outlet = dirichletbc(PETSc.ScalarType(0), locate_dofs_topological(Q, fdim, ft.find(outlet_marker)), Q)
    bcp = [bcp_outlet]

    bcs = [bcu_inlet, bcu_wall, bcu_obstacle, bcp_outlet] 
    
    
    
    # Assemble matrices and creating nullspace                               
    def  block_operators():
        A = assemble_matrix_block(a, bcs=bcu)
        A.assemble()  
        b = assemble_vector_block(L, a, bcs=bcs) 

        return A, b
    
    
    
    # LU MUMPS solver Solve the Stokes problem using blocked matrices and a direct solver.
    def block_direct_solver():
    
        # Assembler the block operator and RHS vector
        A, b = block_operators()

        # Create a solver
        ksp = PETSc.KSP().create(mesh.comm)
        ksp.setOperators(A)
        ksp.setType("preonly")

        # Set the solver type to MUMPS (LU solver) and configure MUMPS to
        # handle pressure nullspace
        pc = ksp.getPC()
        pc.setType("lu")
        sys = PETSc.Sys()  # type: ignore
        use_superlu = PETSc.IntType == np.int64
        if sys.hasExternalPackage("mumps") and not use_superlu:
            pc.setFactorSolverType("mumps")
            pc.setFactorSetUpSolverType()
            pc.getFactorMatrix().setMumpsIcntl(icntl=24, ival=1)
            pc.getFactorMatrix().setMumpsIcntl(icntl=25, ival=0)
        else:
            pc.setFactorSolverType("superlu_dist")

        # Create a block vector (x) to store the full solution, and solve
        x = A.createVecLeft()
        ksp.solve(b, x)

        # Create Functions and scatter x solution
        u, p = Function(V), Function(Q)
        offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
        u.x.array[:offset] = x.array_r[:offset]
        p.x.array[: (len(x.array_r) - offset)] = x.array_r[offset:]

        return u, p


    
    # solving the system
    u_solution, p_solution = block_direct_solver() 
    
    

    #L2 norms of the solution fields
    L2_def_u = form(inner(u_solution, u_solution) * ufl.dx)
    L2_norm_u = np.sqrt(mesh.comm.allreduce(assemble_scalar(L2_def_u), op=MPI.SUM))
    

    L2_def_p = form(inner(p_solution, p_solution) * ufl.dx)
    L2_norm_p = np.sqrt(mesh.comm.allreduce(assemble_scalar(L2_def_p), op=MPI.SUM))
    
    velocity_norm.append(L2_norm_u)
    pressure_norm.append(L2_norm_p)
    
    if MPI.COMM_WORLD.rank == 0:
        print(f"Norm of u: {L2_norm_u}")
        print(f"Norm of p: {L2_norm_p}")  
        


    #Mesh discretization parameters
    num_cells_local = mesh.topology.index_map(gdim).size_local
    print(f"Number of cells: {num_cells_local}")
    total_number_of_cells.append(num_cells_local)

    cell_indices = np.arange(num_cells_local, dtype=np.int32)
    h_local = dolfinx.cpp.mesh.h(mesh._cpp_object, gdim, cell_indices)

    #Minimum element size:
    hmin_local = np.min(h_local)
    hmin = mesh.comm.allreduce(hmin_local, op=MPI.MIN)
    min_element_size.append(hmin)
    print(f"Minimum element size: {hmin}")

    #Maximum element size:
    hmax_local = np.max(h_local)
    hmax = mesh.comm.allreduce(hmax_local, op=MPI.MAX)
    max_element_size.append(hmax)
    print(f"Maximum element size: {hmax}")
    
    
    V_Dofs = V.dofmap.index_map.size_global
    P_Dofs = Q.dofmap.index_map.size_global
    tot_Dofs = V_Dofs + P_Dofs
    total_number_of_Dofs.append(tot_Dofs)

    print(f"Number velocity of degrees of freedom:{V_Dofs:,}")
    print(f"Number pressure of degrees of freedom:{P_Dofs:,}")
    print(f"Total number of degrees of freedom:{tot_Dofs:,}")
    





    # Pressure difference
    tree = bb_tree(mesh, mesh.geometry.dim)
    points = np.array([[0.15, 0.2, 0], [0.25, 0.2, 0]])
    cell_candidates = compute_collisions_points(tree, points)
    colliding_cells = compute_colliding_cells(mesh, cell_candidates, points)
    front_cells = colliding_cells.links(0)
    back_cells = colliding_cells.links(1)
    
    p_front = None
    if len(front_cells) > 0:
        p_front = p_solution.eval(points[0], front_cells[:1])
    p_front = mesh.comm.gather(p_front, root=0)
    p_back = None
    if len(back_cells) > 0:
        p_back = p_solution.eval(points[1], back_cells[:1])
    p_back = mesh.comm.gather(p_back, root=0)

    
    if mesh.comm.rank == 0:
        # Choose first pressure that is found from the different processors
        for pressure in p_front:
            if pressure is not None:
                p_diff[refinement_level] = pressure[0]
                break
        for pressure in p_back:
            if pressure is not None:
                p_diff[refinement_level] -= pressure[0]
                break
            
    print(f"Pressure difference: {p_diff[refinement_level]}")



    # physical quantities
    mu = Constant(mesh, PETSc.ScalarType(0.1))   # Dynamic viscosity
    rho = Constant(mesh, PETSc.ScalarType(1))     # Density
    dObs = Measure("ds", domain=mesh, subdomain_data=ft, subdomain_id=obstacle_marker)
    n = -FacetNormal(mesh)  # Normal vector pointing out of obstacle

    # Drag and lift coefficients
    u_t = inner(as_vector((n[1], -n[0])), u_solution)
    drag = form(0.002 * (mu / rho * inner(grad(u_t), n) * n[1] - p_solution * n[0]) * dObs)
    lift = form(-0.002* (mu / rho * inner(grad(u_t), n) * n[0] + p_solution * n[1]) * dObs)
    

    drag_coeff = mesh.comm.gather(assemble_scalar(drag), root=0)
    lift_coeff = mesh.comm.gather(assemble_scalar(lift), root=0)


    if MPI.COMM_WORLD.rank == 0:
        drag_coefficient.append(drag_coeff[0])
        lift_coefficient.append(lift_coeff[0])
        
        print(f"Drag Coefficient: {drag_coeff[0]}")
        print(f"Lift Coefficient: {lift_coeff[0]}")
        

    
    # Calculate CPU time
    cpu_time = time.process_time() - start_time
    
    # memory usage after solving
    mem_after = process.memory_info().rss / (1024 * 1024) 
    mem_used = mem_after - mem_before
    
    print(f"CPU time: {cpu_time:.2f} seconds")
    print(f"Memory usage, Δ: {mem_used:.2f} MB")

    
    
    # saving the solution
    # Subfolder per refinement level
    output_folder = base_output / f"refinement_{refinement_level}"
    output_folder.mkdir(parents=True, exist_ok=True)

    # .bp filenames for ADIOS2
    velocity_bp = base_output / f"velocity_level_{refinement_level}.bp"
    pressure_bp = base_output / f"pressure_level_{refinement_level}.bp"
    
    # Save VTK (.pvd/.vtu)
    velocity_vtk = output_folder / "velocity.pvd"
    pressure_vtk = output_folder / "pressure.pvd"
    

    # Save .bp files
    with VTXWriter(mesh.comm, str(velocity_bp), [u_solution]) as vtx:
        vtx.write(float(refinement_level))

    with VTXWriter(mesh.comm, str(pressure_bp), [p_solution]) as vtx:
        vtx.write(float(refinement_level))

    # Save .pvd/.vtu
    with VTKFile(mesh.comm, str(velocity_vtk), "w") as vfile:
        vfile.write([u_solution._cpp_object], float(refinement_level))

    with VTKFile(mesh.comm, str(pressure_vtk), "w") as pfile:
        pfile.write([p_solution._cpp_object], float(refinement_level))
    

    with open(log_filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            refinement_level, res_min, res_max, p_diff[refinement_level], L2_norm_u, L2_norm_p,
            hmin, hmax, V_Dofs, P_Dofs, tot_Dofs, num_cells_local,
            drag_coefficient[refinement_level], lift_coefficient[refinement_level],
            cpu_time, mem_used
        ])
        
    gmsh.model.mesh.clear()
    gmsh.model.occ.synchronize()
    
    



















# mesh visualization
def plot_mesh(mesh: Mesh, values=None):
       
    # We create a pyvista plotter instance
    plotter = pyvista.Plotter()

    # Since the meshes might be created with higher order elements,
    # we start by creating a linearized mesh for nicely inspecting the triangulation.
    V_linear = functionspace(mesh, ("Lagrange", 1))
    linear_grid = pyvista.UnstructuredGrid(*vtk_mesh(V_linear))

    # If the mesh is higher order, we plot the nodes on the exterior boundaries,
    # as well as the mesh itself (with filled in cell markers)
    if mesh.geometry.cmap.degree > 1:
        ugrid = pyvista.UnstructuredGrid(*vtk_mesh(mesh))
        if values is not None:
            ugrid.cell_data["Marker"] = values
        plotter.add_mesh(ugrid, style="points", color="b", point_size=10)
        ugrid = ugrid.tessellate()
        plotter.add_mesh(ugrid, show_edges=False)
        plotter.add_mesh(linear_grid, style="wireframe", color="black")
    else:
        # If the mesh is linear we add in the cell markers
        if values is not None:
            linear_grid.cell_data["Marker"] = values
        plotter.add_mesh(linear_grid, show_edges=True)

    # We plot the coordinate axis and align it with the xy-plane
    plotter.show_axes()
    plotter.view_xy()
    plotter.show()

plot_mesh(mesh)






"""
# Other visualizations 
# After the refinement loop, plot the results
if len(total_number_of_Dofs) > 0 and len(min_element_size) > 0 and len(total_number_of_Dofs) == len(min_element_size):
    plt.loglog(total_number_of_Dofs, min_element_size, marker='o', label='Minimum Element Size')
else:
    print("Warning: Incompatible or empty data for minimum element size plot.")

if len(total_number_of_Dofs) > 0 and len(max_element_size) > 0 and len(total_number_of_Dofs) == len(max_element_size):
    plt.loglog(total_number_of_Dofs, max_element_size, marker='x', label='Maximum Element Size')
else:
    print("Warning: Incompatible or empty data for maximum element size plot.")

plt.xlabel('Degrees of Freedom')
plt.ylabel('Element Size')
plt.legend()
plt.grid(True)
plt.title('Element Size vs. Degrees of Freedom')
plt.show()



# 1. Drag Coefficient
plt.figure()
plt.plot(min_element_size, drag_coefficient, "bo-", label="Drag Coefficient")
plt.xlabel("Mesh size h")
plt.ylabel("Drag Coefficient")
plt.title("Convergence of Drag Coefficient")
plt.grid(True, which="both", ls="--")
plt.legend()
plt.tight_layout()
plt.gca().invert_xaxis()
plt.show()

# 2. Lift Coefficient
plt.figure()
plt.plot(min_element_size, lift_coefficient, "gs-", label="Lift Coefficient")
plt.xlabel("Mesh size h")
plt.ylabel("Lift Coefficient")
plt.title("Convergence of Lift Coefficient")
plt.grid(True, which="both", ls="--")
plt.legend()
plt.tight_layout()
plt.gca().invert_xaxis()
plt.show()



# 1. Drag Coefficient
plt.figure()
plt.plot(iterations, drag_coefficient, marker='o', color='blue')
plt.xlabel("Refinement Iteration")
plt.ylabel("Drag Coefficient")
plt.title("Drag Coefficient vs Refinement Iteration")
plt.grid(True)
plt.tight_layout()
plt.show()


# 2. Lift Coefficient
plt.figure()
plt.plot(iterations, lift_coefficient, marker='s', color='green')
plt.xlabel("Refinement Iteration")
plt.ylabel("Lift Coefficient")
plt.title("Lift Coefficient vs Refinement Iteration")
plt.grid(True)
plt.tight_layout()
plt.show()



# 3. Velocity Norm
plt.figure()
plt.plot(iterations, velocity_norm, marker='^', color='red')
plt.xlabel("Refinement Iteration")
plt.ylabel("Velocity Norm")
plt.title("Velocity Norm vs Refinement Iteration")
plt.grid(True)
plt.tight_layout()
plt.show()

# 4. Pressure Norm
plt.figure()
plt.plot(iterations, pressure_norm, marker='d', color='purple')
plt.xlabel("Refinement Iteration")
plt.ylabel("Pressure Norm")
plt.title("Pressure Norm vs Refinement Iteration")
plt.grid(True)
plt.tight_layout()
plt.show()

# 5. Pressure Difference
plt.figure()
plt.plot(iterations, p_diff, marker='x', color='orange')
plt.xlabel("Refinement Iteration")
plt.ylabel("Pressure Difference")
plt.title("Pressure Difference vs Refinement Iteration")
plt.grid(True)
plt.tight_layout()
plt.show()






# solution visualization
# Velocity field
topology, cell_types, geometry = vtk_mesh(V)
values = np.zeros((geometry.shape[0], 3), dtype=np.float64)
values[:, :len(u_solution)] = u_solution.x.array.real.reshape((geometry.shape[0], len(u_solution)))

# Create a point cloud of glyphs
function_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
function_grid["u"] = values
glyphs = function_grid.glyph(orient="u", factor=0.1)

# Create a pyvista-grid for the mesh
mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim)
grid = pyvista.UnstructuredGrid(*vtk_mesh(mesh, mesh.topology.dim))

# Create plotter
glyph_plotter = pyvista.Plotter()
glyph_plotter.add_mesh(grid, style="wireframe", color="k")
glyph_plotter.add_mesh(glyphs)
glyph_plotter.view_xy()
glyph_plotter.show()  


# Velocity magnitude
# Create a pyvista-grid for the mesh
velocity_magnitude = np.linalg.norm(values, axis=1)
function_grid["velocity_magnitude"] = velocity_magnitude

# Create plotter
plotter = pyvista.Plotter()
plotter.add_mesh(function_grid, scalars="velocity_magnitude", cmap="plasma", show_edges=False)
plotter.view_xy()
plotter.show()


# Pressure contour plot
topology, cell_types, geometry = vtk_mesh(Q)  # Assuming Q is for pressure
values = p_solution.x.array  # Pressure values

# Create PyVista UnstructuredGrid
pressure_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
pressure_grid["Pressure"] = values  # Assign pressure values

# Create a contour plot
pressure_plotter = pyvista.Plotter()
pressure_plotter.add_mesh(pressure_grid, cmap="plasma", show_edges=False)
pressure_plotter.view_xy()
pressure_plotter.show()
"""
