import matplotlib.pyplot as plt
import numpy as np
import math

def get_lines(poly):
    lines = []
    for x in range(len(poly)-1):
        lines.append((poly[x],poly[x+1]))
    lines.append((poly[len(poly)-1],poly[0]))
    return lines

def get_angle(p):
    x,y = p
    return (math.atan2(y,x) + math.pi)/(2 * math.pi)

def dist(p):
    x,y = p
    return math.sqrt(x**2+y**2)

def interpolate(p1,p2,angle):
    angle += 1e-8
    x1,y1 = p1
    x2,y2 = p2
    dx,dy = x1-x2, y1-y2

    tana = math.tan(angle)
    t = (-y1+x1*tana)/(dy-dx*tana)
    l = (t*dy+y1)/math.sin(angle)
    point = (t*dx+x1,t*dy+y1)

    return dist(point)

def trace_polygon(poly,num_points):
    lines = get_lines(poly)
    div_binning = 1./num_points
    bin_dists = [None]*num_points
    for p1,p2 in lines:
        ang_start = get_angle(p1)
        ang_end = get_angle(p2)
        tot_diff = (ang_end - ang_start) % 1.
        ang_end = ang_start + tot_diff
        bin_start = ang_start + (-ang_start % div_binning)
        bin_num = int(ang_start/div_binning-1e-7)
        bin_pos = bin_start
        while bin_pos < ang_end:
            bin_dists[bin_num] = interpolate(p1,p2,bin_pos*2*math.pi)
            bin_num = (bin_num+1)%num_points
            bin_pos += div_binning
    return bin_dists

if __name__ == "__main__":
    line1 = [(-1,-5),(0,-10),(5,0),(1,1),(2,2),(-2,0)]
    num_points = 64
    dists = trace_polygon(line1,num_points)
    #print(get_angle((-0.0001,1)))
    points = []
    for i,dist in enumerate(dists):
        angle = (-get_angle(line1[0])+0.75+(i / num_points)) * 2 * math.pi
        x = math.cos(angle)*dist
        y = math.sin(angle)*dist
        points.append((x,y))

    ps = np.transpose(np.array(line1))
    plt.plot(ps[0],ps[1] , marker='o', markerfacecolor='blue', markersize=4, color='skyblue', linewidth=2)
    ps2 = np.transpose(np.array(points))
    plt.plot(ps2[0],ps2[1] , marker='o', markerfacecolor='red', markersize=4, color='red', linewidth=2)
    plt.show()
