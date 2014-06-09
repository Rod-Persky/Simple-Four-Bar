from math import cos, sin, atan, atan2, pi, sqrt
from cmath import phase
from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np

# First set up the figure, the axis, and the plot element we want to animate
fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex=True)

ax    = plt.axes(xlim=(-3, 13), ylim=(-5, 10))
line,  = ax.plot([], [], lw=2)
wedge1, = ax.plot([], [], lw=2)
ani_frames = 1000
ax.axis('off')

t2_range = [112.7, 180+67.5869]
w2 = 15
a2 = 0

show_text = False
going_down = False

#wedge shape        A       P      Y      X      B      A
wedge1_vertexes  = [[0.906, 7.547, 8.516, 7.984, 8.250, 0.906],
                   [3.609,  2.250, 2.234, 0.641,-0.391, 3.609]
                   ]

wedge2_vertexes = [[-1.766, 4.859, 5.719, 5.797, 6.563, -1.766],
                   [5.984,  7.188, 7.516, 6.031, 5.047,  5.984]
                   ]
   
wedge3_vertexes = [[-3.922,-0.172, 0.141, 1.234, 2.484, -3.922],
                   [ 2.563, 8.234, 9.047, 8.078, 7.969,  2.563]
                   ]
                   

def graphical_method(point):
    # Calculate center of rotation and length from point A and B
    A = np.array(point)
    dist_a12      = sqrt((A[1][0]-A[0][0])**2 + (A[1][1]-A[0][1])**2)
    midpoint_a12  = (A[1] + A[0])/2
    gradient_a12p = -(A[1][0]-A[0][0])/(A[1][1]-A[0][1])
    constant_a12p = midpoint_a12[1] - (gradient_a12p * midpoint_a12[0])
    
    dist_a23      = sqrt((A[2][0]-A[1][0])**2 + (A[2][1]-A[1][1])**2)
    midpoint_a23  = (A[2] + A[1])/2
    gradient_a23p = -(A[2][0]-A[1][0])/(A[2][1]-A[1][1])
    constant_a23p = midpoint_a23[1] - (gradient_a23p * midpoint_a23[0])
    
    xint_a12a23 = (constant_a23p - constant_a12p)/(gradient_a12p - gradient_a23p)
    yint_a12a23 = xint_a12a23*gradient_a12p + constant_a12p  
    #ax.plot(xint_a12a23, yint_a12a23, 'ro')
    
    link_length = sqrt((A[1][0] - xint_a12a23)**2 + (A[1][1] - yint_a12a23)**2)
    print("link length =", link_length)
    print("position    =", [xint_a12a23, yint_a12a23])
    return [xint_a12a23, yint_a12a23], link_length


# Next for our variables
# First, lets get the data to calculate the centeres of rotation from the 3
#  wedge locations
point_a = [[wedge1_vertexes[0][0], wedge1_vertexes[1][0]],
           [wedge2_vertexes[0][0], wedge2_vertexes[1][0]],
           [wedge3_vertexes[0][0], wedge3_vertexes[1][0]]
          ]

point_b = [[wedge1_vertexes[0][4], wedge1_vertexes[1][4]],
           [wedge2_vertexes[0][4], wedge2_vertexes[1][4]],
           [wedge3_vertexes[0][4], wedge3_vertexes[1][4]]
          ]

# Calculate lengths from given information
point_a, a = graphical_method(point_a)
point_b, c = graphical_method(point_b)
#b = sqrt((wedge1_vertexes[1][0]-wedge1_vertexes[1][4])**2 + (wedge1_vertexes[0][0]-wedge1_vertexes[0][4])**2)
#b = sqrt((wedge2_vertexes[1][0]-wedge2_vertexes[1][4])**2 + (wedge2_vertexes[0][0]-wedge2_vertexes[0][4])**2)
b = sqrt((wedge3_vertexes[1][0]-wedge3_vertexes[1][4])**2 + (wedge3_vertexes[0][0]-wedge3_vertexes[0][4])**2)


d = sqrt((point_a[1]-point_b[1])**2 + (point_a[0]-point_b[0])**2)

# Next for our variables
l1 = d #= 2.503
l2 = a #= 3.788
l3 = b #= 8.382
l4 = c #= 8.408


#l1 = d = 3.789
#l2 = a = 2.509
#l3 = b = 8.363
#l4 = c = 8.406


def init():
    line.set_data([], [])
    wedge1.set_data([], [])
    return line, wedge1

def rotate_vertex_list(vertexes, angle):
    delta_angle  = angle - (t2_range[0])*pi/180 
    new_vertexes = [[vertexes[0][0]],[vertexes[1][0]]]
    origin       = [vertexes[0][0],   vertexes[1][0]]
    
    for i in xrange(1, len(vertexes[1])):
        vertex    = [vertexes[0][i], vertexes[1][i]]
        m_origin  = sqrt(pow(vertex[0]-origin[0],2) + pow(vertex[1]-origin[1],2))
        angle_ori = atan2(vertex[1]-origin[1], vertex[0]-origin[0])
        new_vertexes[0].append(origin[0] + m_origin*cos(angle_ori + delta_angle))
        new_vertexes[1].append(origin[1] + m_origin*sin(angle_ori + delta_angle))

    return new_vertexes

def calculate_positions(time):
    # Convert to radians
    global going_down
    
    if time==0:
        if going_down:
            going_down = False
        else:
            going_down = True

    if not going_down:
        t2 = (t2_range[1] - (time/float(ani_frames))*(t2_range[1]-t2_range[0])) * pi / 180
    else:
        t2 = ((time/float(ani_frames))*(t2_range[1]-t2_range[0])+t2_range[0]) * pi / 180

    # Calculate position
    k1 = d / a
    k2 = d / c
    k3 = (a*a - b*b + c*c + d*d)/(2*a*c)
    k4 = d/b
    k5 = (c*c - d*d - a*a - b*b)/(2*a*b)

    A = cos(t2) - k1 - k2*cos(t2) + k3
    B = -2 * sin(t2)
    C = k1 - (k2+1)*cos(t2) + k3
    D = cos(t2) - k1 + k4*cos(t2) + k5
    E = -2*sin(t2)
    F = k1 + (k4-1)*cos(t2) + k5

    t3_open = t3 = 2*atan((-E - sqrt(E*E - 4*D*F))/(2*D))
    t3_clos = 2*atan((-E + sqrt(E*E - 4*D*F))/(2*D))
    t4_open = t4 = 2*atan((-B - sqrt(B*B - 4*A*C))/(2*A))
    t4_clos = 2*atan((-B + sqrt(B*B - 4*A*C))/(2*A))

    if show_text:
        print "k   =", [ '%.3f' % elem for elem in [k1, k2, k3, k4, k5]]
        print "A.. =", [ '%.3f' % elem for elem in [A, B, C, D, E, F]]
        print "t3  =", [ '%.3f' % elem for elem in [t3_open*180/pi, t3_clos*180/pi]]
        print "t4  =", [ '%.3f' % elem for elem in [t4_open*180/pi, t4_clos*180/pi]]
        print "The angles (assuming the system is open) are:"
        print "theta 1 = 0"
        print "theta 2 = {:.3f}".format(t2*180/pi)
        print "theta 3 = {:.3f}".format(t3_open*180/pi)
        print "theta 4 = {:.3f}".format(t4_open*180/pi)

    # Calculate velocity
    w3 = ((a * w2)/b) * sin(t4-t2)/sin(t3-t4)
    w4 = ((a * w2)/c) * sin(t2-t3)/sin(t4-t3)

    if show_text:
        print "\nThe rotational velocity components are:"
        print "omega 1 = 0"
        print "omega 2 = {:.3f}".format(w2)
        print "omega 3 = {:.3f}".format(w3)
        print "omega 4 = {:.3f}".format(w4)

    va  = a*w2*(-sin(t2) + 1j*cos(t2))
    vba = b*w3*(-sin(t3) + 1j*cos(t3))
    vb  = c*w4*(-sin(t4) + 1j*cos(t4))

    
    if show_text:
        print "\nThe velociy components are:"
        for v in zip(["va", "vba", "vb"],[va, vba, vb]):
            print "{:3s}: x = {:7.3f}, y = {:7.3f}, mag = {:7.3f}, ang = {:7.3f}".format(v[0], v[1].real, v[1].imag, abs(v[1]), phase(v[1])*180/pi)

    # Calculate acceleration
    A = c*sin(t4)
    B = b*sin(t3)
    C = a*a2*sin(t2) + a*w2*w2*cos(t2) + b*w3*w3*cos(t3) - c*w4*w4*cos(t4)
    D = c*cos(t4)
    E = b*cos(t3)
    F = a*a2*cos(t2) - a*w2*w2*sin(t2) - b*w3*w3*sin(t3) + c*w4*w4*sin(t4)

    a3 = (C*D - A*F)/(A*E - B*D)
    a4 = (C*E - B*F)/(A*E - B*D)

    aax  = -a*a2*sin(t2) - a*w2*w2*cos(t2)
    aay  =  a*a2*cos(t2) - a*w2*w2*sin(t2)
    abax = -b*a3*sin(t3) - b*w3*w3*cos(t3)
    abay =  b*a3*cos(t3) - b*w3*w3*sin(t3)
    abx  = -c*a4*sin(t4) - c*w4*w4*cos(t4)
    aby  =  c*a4*cos(t4) - c*w4*w4*sin(t4)

    if show_text:
        print "\nAcceleration:"
        print "alpha 3 = {:.3f}".format(a3)
        print "alpha 4 = {:.3f}".format(a4)
        print "A.. =", [ '%.3f' % elem for elem in [A, B, C, D, E, F]]
        print "al. =", [ '%.3f' % elem for elem in [a3, a4]]
        print "\nAcceleration components:"
        for A in zip(["Aa", "Aba", "Ab"], [[aax, aay], [abax, abay], [abx, aby]]):
            print "{0:4s}x = {1:9.3f} y = {2:7.3f}".format(A[0], A[1][0], A[1][1])

    # Calculate location of links
    #                   X,         Y
    vertex_a = [0,0]
    vertex_b = [a*cos(t2) + vertex_a[0], a*sin(t2) + vertex_a[1]]
    vertex_c = [b*cos(t3) + vertex_b[0], b*sin(t3) + vertex_b[1]]
    vertex_d = [d + vertex_a[0], vertex_a[1]] 

    vertexs_x = [elem[0] for elem in [vertex_d, vertex_a, vertex_b, vertex_c, vertex_d]]
    vertexs_y = [elem[1] for elem in [vertex_d, vertex_a, vertex_b, vertex_c, vertex_d]]
    vertexs = [vertexs_x, vertexs_y]
    vertexs = rotate_vertex_list(vertexs, 0*pi/180)

    # Setup wedge translation
    wedge1_vertexes_current = [[],[]]
    corr = [wedge1_vertexes[0][0]-vertexs[0][2], wedge1_vertexes[1][0]-vertexs[1][2]]
    wedge1_vertexes_current[0] = [x-corr[0] for x in wedge1_vertexes[0]]
    wedge1_vertexes_current[1] = [y-corr[1] for y in wedge1_vertexes[1]]
    wedge1_vertexes_current = rotate_vertex_list(wedge1_vertexes_current, t3+(29*pi/180))
    wedge1.set_data(wedge1_vertexes_current[0], wedge1_vertexes_current[1])

    line.set_data(vertexs[0], vertexs[1])
    return line, wedge1

#calculate_positions(0)
anim = animation.FuncAnimation(fig, calculate_positions, init_func=init, frames=ani_frames, interval=20, blit=True)
#anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
plt.show()
