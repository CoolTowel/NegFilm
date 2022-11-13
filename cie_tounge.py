import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as ln

def XYZ2xyY(XYZ):
    XYZ = np.array(XYZ)
    XYZ_sum = np.sum(XYZ,axis=-1)
    xyz = XYZ/XYZ_sum[:,None]
    xyY = np.copy(xyz)
    xyY[:,2] = XYZ[:,1]
    return xyY

T_srgb_to_cie=np.array([[0.4124,0.3576,0.1805],
                    [0.2126,0.7152,0.0722],
                    [0.0193,0.1192,0.9505]])

def srgb2cieXYZ(srgb):
    T_srgb_to_cie=np.array([[0.4124,0.3576,0.1805],
                       [0.2126,0.7152,0.0722],
                       [0.0193,0.1192,0.9505]])
    srgb = np.array(srgb)
    XYZ = np.dot(T_srgb_to_cie, srgb.T)
    
def srgb2ciexyY(srgb):
    T_srgb_to_cie=np.array([[0.4124,0.3576,0.1805],
                       [0.2126,0.7152,0.0722],
                       [0.0193,0.1192,0.9505]])
    srgb = np.array(srgb)
    XYZ = np.dot(T_srgb_to_cie, srgb.T)
    XYZ_sum = np.sum(XYZ)
    x = XYZ[0]/XYZ_sum
    y = XYZ[1]/XYZ_sum
    Y = XYZ[1]
    xyY = np.array([x,y,Y]).T
    return xyY

def ciexyY2srgb(ciexyY):
    T_cie_to_srgb = ln.inv(T_srgb_to_cie)
    ciexyY = np.array(ciexyY)
    XYZ = np.copy(ciexyY)
    XYZ[:,0] = ciexyY[:,2]/ciexyY[:,1]*ciexyY[:,0]
    XYZ[:,1] = ciexyY[:,2]
    XYZ[:,2] = ciexyY[:,2]/ciexyY[:,1]*(1-ciexyY[:,0]-ciexyY[:,1])
    srgb = np.dot(T_cie_to_srgb, XYZ.T).T

    return srgb

def _srgb_gamma(srgb):
    if srgb<=0: #gamut clip 
        return 0
    else:
        a = 0.055
        gamma = 2.4
        if srgb<0.00304:
            srgb = 12.92*srgb
        else:
            srgb = (1+a)*srgb**(1/gamma)-a

        # srgb = int(255*np.nan_to_num(srgb_gamma))
        return srgb

srgb_gamma_8bit = np.vectorize(_srgb_gamma)

cmf_XYZ = np.genfromtxt('cc2012xyz2_1_5dp.csv', delimiter=',')
cmf_xyY = np.hstack((cmf_XYZ[:,0].reshape(-1,1), XYZ2xyY(cmf_XYZ[:,1:])))

res = 400
xx, yy = np.meshgrid(np.linspace(0.00001,1,res),np.linspace(0.00001,1,res))
all_color_matrix = np.dstack((xx, yy)) # [x,y] in matrix
all_color_list = all_color_matrix.reshape(res**2,2)
Y = np.ones((res**2,1)) # identical Y(luminance) for all color

all_color_list = np.hstack((all_color_list,Y)) # color in xyY
all_color_list_srgb = ciexyY2srgb(all_color_list)

all_color_matrix = all_color_list.reshape(res,res,3) 

all_color_matrix_srb =all_color_list_srgb.reshape(res,res,3)
locus_srgb_max = np.max(all_color_matrix_srb,axis=-1) 
all_color_matrix_srb = all_color_matrix_srb/locus_srgb_max[:,:,None]
all_color_matrix_srb_gamma_8bit = srgb_gamma_8bit(all_color_matrix_srb)

spectrum_locus_coord = cmf_xyY[:,1:3] #xy
spectrum_locus_coord = np.vstack((spectrum_locus_coord,spectrum_locus_coord[0]))

path_codes =  np.ones(len(spectrum_locus_coord),
                              dtype=mpath.Path.code_type) * mpath.Path.LINETO
path_codes[0] = mpath.Path.MOVETO
spectrum_locus_path = mpath.Path(spectrum_locus_coord, path_codes)
spectrum_locus_patch = mpatches.PathPatch(spectrum_locus_path,
                                        facecolor=[0.3, 0.3, 0.3, 0.0],
                                        linewidth=1,
                                        zorder=1000)

fig, ax = plt.subplots()
# ax.plot(cmf_xyY[:,1],cmf_xyY[:,2])
ax.set_aspect('equal', adjustable='box')
locus_im = ax.imshow(all_color_matrix_srb[::-1], extent=(0, 1, 0, 1))
locus_im.set_clip_path(spectrum_locus_path, transform = ax.transData)
ax.add_patch(spectrum_locus_patch)
plt.show()