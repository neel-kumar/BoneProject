from solid import *
from solid.utils import *
import subprocess

files = input('scad file: ') or '../scad/voxel5.scad'
subprocess.run([r'C:\Program Files\OpenSCAD\openscad.exe', '-o', 'out.stl', './' + files])

