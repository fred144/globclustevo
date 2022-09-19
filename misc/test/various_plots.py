# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------#
#
# Description:
#   いろいろなプロットを作成
#
# Example:
#  - 全領域のプロット #All area plot
#    python various_plots.py workdir -dd datadir -step 1
#  - ハロー周辺のプロット # Plot around halo
#    python various_plots.py workdir -dd datadir -step 1 -obj halo -wl "3 30"
#  - ハローvirial半径内の密度最大点周辺のプロット #Plot around the maximum density point within the halo virial radius
#    python various_plots.py workdir -dd datadir -step 1 -obj dmax -wl "3 30"
#  - 星周辺のプロット #Plot around the star
#    python various_plots.py workdir -dd datadir -step 1 -obj star -wl "3 30"
#
# Input:
#      workdir: working directory
#      -step step: プロットするデータのステップ数 
#      -dd datadir: data directory (assumed to be the same as workdir in default)
#      -obj object_type: object type (halo or star)
#      -id object_id: object id (mass order for halo; birth redshift order for star)
#      --DM_only: skip hydro shapshots
#
#------------------------------------------------------------------------------#


#---------------------------------------------------#
#        environment dependent configuration        #
#---------------------------------------------------#
#
from socket import gethostname
#
# ホスト名から環境を推定
#
hostname=gethostname()
print ("hostname = "+hostname, flush=True)

#
#draco
#
if 'draco' in hostname: #dracoという文字列がhostnameに含まれれば
    print ("\n*** configuration for draco used ***\n", flush=True)
    DWARFDIR = '~/ramses/dwarf'
    PYTHON = '~/local/bin/python'
    import matplotlib           
    matplotlib.use('Agg')     #$DISPLAY云々のエラーを回避
#
#an
#
elif 'an' in hostname: #anという文字列がhostnameに含まれれば
    print ("\n*** configuration for an used ***\n", flush=True)
    DWARFDIR = '~/ramses/dwarf'
    PYTHON = '/opt/local/Anaconda/anaconda2/bin/python'
    import matplotlib           
    matplotlib.use('Agg')     #$DISPLAY云々のエラーを回避
#
#その他 (MBPを想定)
#
else:
    print ("\n*** configuration for MBP or others used ***\n", flush=True)
    DWARFDIR = '~/projects/dwarf'
    PYTHON = 'python2.7'
    import matplotlib
#---------------------------------------------------#
#---------------------------------------------------#

#-------- imoprt libraies --------#

import numpy as np
import sys
import glob
import os
import argparse
import matplotlib.pyplot as plt
import time
import yt
import shutil

print ('calling: ', ' '.join(sys.argv), flush=True)

#from yt.extensions.astro_analysis.halo_analysis.api import HaloCatalog
#上の書き方だとp.annotate_halo()でエラーが発生、plot_modifications.py内でis_instanceの書き方に不整合？
from yt.extensions.astro_analysis.halo_analysis.halo_catalog import HaloCatalog
#from yt.extensions.attic.halo_mass_function.api import *

#local library
sys.path.append(os.path.expanduser(DWARFDIR+'/python/lib'))

#scale factorのtableを取得するための関数
from get_scale_factor_table import get_scale_factor_table

#------------------------------   paser  ----------------------------#
parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description='''general analysis of Ramses data at given step'''
)
parser.add_argument('workdir',help='working directory')
parser.add_argument('-step',help='step of files to be converted', default='-1', required=True)
parser.add_argument('-dd', '--datadir',  type=str, default='',
                    help='name of data directory (default: same as workdir)')
parser.add_argument('-obj', '--object_type',  type=str, default='sim',
                    help='object type (sim, halo, dmax, disk or star)')
parser.add_argument('-id', '--object_id',  type=int, default=1,
                    help='object id (mass order for halo; idp for star)')
parser.add_argument('-wl', '--width_list',  type=str, default='whole',
                    help='list for the width of plot boxsize (length in kpc; or whole)')
parser.add_argument('--DM_only', action='store_true',
                    help='skip hydro shapshots')
parser.add_argument('-pvs', '--plot_variables',type=str, default='nH T nT',
                    help='variables to be plotted (either nH,T,nHproj,Tproj,yH2,yHII,Sigma,phi,phiproj,pDM,DMhalo,pStar,Mach,metal,sStar,sDM,nT,MachT,nmetal,nyH2,nyHII,metalyH2,MF,J21)')
parser.add_argument('-sm', '--star_marker',type=str, 
                    help='marker for star annotation (either cross or dot)')
parser.add_argument('-ax', '--plot_axis',type=str, default='xyz',
                    help='axis of snapshot planes (eighter xyz (3 panels), x, y, or z)')
parser.add_argument('-vmin',type=float, default=None,
                    help='upper limit of plot')
parser.add_argument('-vmax',type=float, default=None,
                    help='lower limit of plot')
#parser.add_argument('-nT_nth',type=float, default=None,
parser.add_argument('-nT_nth',type=float, default=5.458e4,      #for haloD, 5.458e4 for lmax = 25 and 8.528e2 for lmax = 22 
                    help='normalization for star-formation threshold curve in nT plane (nth [cm-3]/ [((1+z)/10)^2 * (T2/100K)])')
parser.add_argument('-hcen', "--halo_center", default='bary',
                    help='halo center used for plot center: either bary (barycenter) or hf (halo finder)')
parser.add_argument('-bv', "--bulk_vel", 
                    help='bulk_vel = "bvx bvy bvz" [km/s] subtracted from velocity fields')
parser.add_argument("-buff_size", type=int,
                    help='resolutino of DM particle plot on a side (default of this script = 320)')


args = parser.parse_args()


#-------------- initial set-up -------------#

yt.enable_parallelism() #MPI並列を有効化
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#workdir
workdir=os.path.expanduser(args.workdir) #working directory

#datadirを定義
if args.datadir == '':
    datadir=workdir 
else:
    datadir=os.path.expanduser(args.datadir)
datadir=os.path.abspath(datadir)#full pathにしておく    

#解析する天体の種類 (sim, halo or star)
object_type=args.object_type
if object_type not in ["sim", "halo", "dmax", "disk", "star"]:
    print ('object type should be either sim(ulation box), halo, density maximum of halo (dmax), halo with rotation polar axis (disk), or star, but %s is specified\nexiting...'%object_type)
    sys.exit()
#天体のid
i_obj = args.object_id

if object_type == "sim":
    outname_prefix = "sim"
elif object_type == "halo":
    outname_prefix = "halo%d"%i_obj
elif object_type == "dmax":
    outname_prefix = "dmax%d"%i_obj
elif object_type == "disk":
    outname_prefix = "disk%d"%i_obj
elif object_type == "star":
    outname_prefix = "idp%d"%i_obj

#プロットする量
pvars = args.plot_variables.split()
print ("plot variables: {:}".format(pvars))

#consistency check
if args.DM_only:
    for pvar in pvars:
        if pvar in ["nH","nHproj","Sigma","pStar","T","Tproj","Mach","metal","yHI","yHII","yH2","sStar","J21"]:
            print ("DM-only option is not compatible with pvar = {:} in pvars\nexiting...".format(pvar))
            sys.exit()

#plotの軸
plot_axis = args.plot_axis

#最大値・最小値の設定
vmin = args.vmin
if vmin is not None:
    print ("vmin = ", vmin)
vmax = args.vmax
if vmax is not None:
    print ("vmax = ", vmax)

#該当ステップのファイルを取得
step=args.step
#ステップ数の値をチェック
if int(step) < 0:
    print ('non-negative step number should be given (step=%d)\nexiting...',int(step), flush=True)
    exit()

#infoファイル
infofile="output_%05d/info_%05d.txt"%(int(step),int(step))
infofile_fp = os.path.abspath(datadir+"/"+infofile) #full path

if rank == 0:
    print ("\ninfofile (full path):%s\n"%infofile_fp, flush=True)

#ファイルの存在チェック
if not os.path.isfile(datadir+'/'+infofile):
    print ("%s not found\neixting..."%(datadir+'/'+infofile), flush=True)
    sys.exit()

#プロット保存用のサブディレクトリ
anldir=workdir+"/"+"analysis_%05d"%int(step)+"/"+outname_prefix
if rank == 0 and not os.path.isdir(anldir):
    print ("create %s"%anldir, flush=True)
    os.makedirs(anldir) #ディレクトリが無ければ作成
comm.barrier()

if rank == 0:
    print ("move to %s"%anldir, flush=True)
os.chdir(anldir)

if rank == 0:
    print ("plot width list: ",args.width_list.split())


#------------------------------------- read data -----------------------------------#
#セル変数の内容
FIELDS = ["Density",
          "x-velocity", "y-velocity", "z-velocity",
          "Pressure",
          "Metallicity",
          "xHI", "xHII", "xHeII", "xHeIII"]
#粒子変数の内容（追加分）
EPF= [('particle_family', 'b'),      #byte size
      ('particle_tag', 'b'),         #byte size
      ('particle_birth_epoch', 'd'), #double size
      ('particle_metallicity', 'd')] #double size

#平均分子量を考慮した温度の定義
def _my_temperature(field, data):
    #y(i): abundance per hydrogen atom
    XH_RAMSES=0.76 #defined by RAMSES in cooling_module.f90
    YHE_RAMSES=0.24 #defined by RAMSES in cooling_module.f90
    mH_RAMSES=yt.YTArray(1.6600000e-24,"g") #defined by RAMSES in cooling_module.f90
    kB_RAMSES=yt.YTArray(1.3806200e-16,"erg/K") #defined by RAMSES in cooling_module.f90

    dn=data["ramses","Density"].in_cgs()
    pr=data["ramses","Pressure"].in_cgs()
    yHI=data["ramses","xHI"]
    yHII=data["ramses","xHII"]
    yHe = YHE_RAMSES*0.25/XH_RAMSES
    yHeII=data["ramses","xHeII"]*yHe
    yHeIII=data["ramses","xHeIII"]*yHe
    yH2=1.-yHI-yHII
    yel=yHII+yHeII+2*yHeIII
    mu=(yHI+yHII+2.*yH2 + 4.*yHe) / (yHI+yHII+yH2 + yHe + yel)
    return pr/dn * mu * mH_RAMSES / kB_RAMSES

#マッハ数の定義
def _Mach(field, data):
    dn=data["ramses","Density"]
    pr=data["ramses","Pressure"]
    cI2 = pr/dn #等温音速
    vx=data["ramses","x-velocity"]
    vy=data["ramses","y-velocity"]
    vz=data["ramses","z-velocity"]
    v2 = vx**2 + vy**2 + vz**2
    mach = np.sqrt(v2/cI2)
    return mach

#金属度 [Z_solar]の定義
def _Zmetal_in_Zsun(field, data):
    Zmetal=data["ramses","Metallicity"]
    Zmetal_in_Zsun = Zmetal/0.02 #devided by Zsun = 0.02
    return Zmetal_in_Zsun

#フロア付き (phase diagramに利用)
def _Zmetal_in_Zsun_floor(field, data):
    floor_value = 1e-7
    Zmetal=data["ramses","Metallicity"]
    Zmetal_in_Zsun = Zmetal/0.02 #devided by Zsun = 0.02
    Zmetal_in_Zsun[Zmetal_in_Zsun < floor_value] = floor_value #フロアを課す
    return Zmetal_in_Zsun

#化学組成比 y(i) = n(i)/nH
#HI
def _yHI(field, data):
    yHI=data["ramses","xHI"]
    return yHI
#HII
def _yHII(field, data):
    yHII=data["ramses","xHII"]
    return yHII
#H2
def _yH2(field, data):
    yHI=data["ramses","xHI"]
    yHII=data["ramses","xHII"]
    yH2=(1.-yHI-yHII)*0.5
    return yH2
#フロア付き (phase diagramに利用)
def _yH2_floor(field, data):
    floor_value = 1e-8
    yHI=data["ramses","xHI"]
    yHII=data["ramses","xHII"]
    yH2=(1.-yHI-yHII)*0.5
    yH2[yH2 < floor_value] = floor_value #フロアを課す
    return yH2

#J21 (LW intensity in unit of 1e-21 erg s-1 cm-2 Hz-1 sr-1)
#conversion from photon density nLW to J21
# n_nu = unu/<hnu> = 4pi/c Jnu/<hnu>
# -> J21 = 1e21 * c[cm/s]* nLW[cm-3]/4pi  * <hnu> / Delta nu = (where <hnu> = 12.4 eV, Delta nu = 2.4 eV/h)
# see yt/frontends/ramses/fields.py for the definition of data["rt","photon_density_1"] in yt
def _J21(field, data):
    from yt.frontends.ramses.field_handlers import RTFieldFileHandler
    p = RTFieldFileHandler.get_rt_parameters(ds).copy()
    p.update(ds.parameters)
    rt_c_frac = p['rt_c_frac'][0] #fraction of reduced light velocity (at levelmin)
    cgs_c =  2.99792458e10     #light velocity
    cgs_h = 6.62606876e-27     #planck constant
    hnu_ave = 12.4e0           #average energy of LW photons in eV
    Delta_nu = 2.4e0/cgs_h           #energy range of LW photons in eV/h
    NpLW=data["rt","photon_density_1"].to_value(units="cm**-3") * rt_c_frac #convert back to physical (not computational) number density
    J21 = yt.YTArray(1e21 * cgs_c / (4*np.pi) * hnu_ave/Delta_nu * NpLW, '1')
    #print ('conversion factor (nLw -> J21): ', 1e21 * cgs_c / (4*np.pi) * hnu_ave/Delta_nu)
    print ('rt_c_frac = {:} is multiplied again to convert computed photon number density to the physical one'.format(rt_c_frac))
    return J21


###########################
# データの読み込み
###########################
#yt.funcs.mylog.setLevel(1) #log表示レベルの設定

ds = yt.load(infofile_fp, fields=FIELDS, extra_particle_fields=EPF)

if not args.DM_only:
    ds.add_field(("gas","my_temperature"),function=_my_temperature, units="K",sampling_type="cell")
    ds.add_field(("gas","Mach"),function=_Mach, units="1",sampling_type="cell")
    ds.add_field(("gas","Zmetal_in_Zsun"),function=_Zmetal_in_Zsun, units="1",sampling_type="cell")
    ds.add_field(("gas","Zmetal_in_Zsun_floor"),function=_Zmetal_in_Zsun_floor, units="1",sampling_type="cell")
    ds.add_field(("gas","yHI"),function=_yHI, units="1",sampling_type="cell")
    ds.add_field(("gas","yHII"),function=_yHII, units="1",sampling_type="cell")    
    ds.add_field(("gas","yH2"),function=_yH2, units="1",sampling_type="cell")
    ds.add_field(("gas","yH2_floor"),function=_yH2_floor, units="1",sampling_type="cell")
    ds.add_field(("gas","J21"),function=_J21, units="1",sampling_type="cell")

#meta data
reg = ds.unit_registry
code_length = reg.lut['code_length'][0]                #code_length
current_redshift = ds.current_redshift                 #redshift
hubble_constant  = ds.hubble_constant                  #hubble_constant h [100 km/s/Mpc]
omega_m = ds.omega_matter                              #omega matter
omega_l = ds.omega_lambda                              #omega lambda
boxsize_in_cMpc_h = ds.domain_width.in_units('Mpccm/h')[0] #boxsize in [cMpc/h]
boxsize_in_kpc = ds.domain_width.in_units('kpc')[0] #boxsize in [kpc]


#############################################################
# birth_epochからbirth redsfhitを得るためのデータを読み込み
#############################################################
#outputディレクトリ中のnamelistからrt_starの値を取得
#Ramsesではrt_star=.true.だとuse_proper_timeが.true.になって、birth_epochにproper_timeが保存される

#namlistからrt_starの値を取得
#namelist_file = glob.glob(datadir+'/*.nml')[0]
#namelist_file = glob.glob(datadir+'/namelist.nml')[0] #temporary

#namelistの名前を変えつつあるのでそれに対応
if os.path.isfile(datadir+'/namelist.nml'):
    #namelist_file = glob.glob(datadir+'/namelist.nml')[0] #namelist.nmlがあればそれを使う
    namelist_file = datadir+'/namelist.nml' #namelist.nmlがあればそれを使う    
#else:
    #namelist_file = glob.glob(datadir+'/*.nml')[0] #無ければ、とりあえず一つ使ってみる
    
namelist_file = os.path.abspath(datadir+"/output_%05d/namelist.txt"%(int(step))) #<- xcで計算したとき、このファイルは文字化け（？）して読み取れない
f = open(namelist_file, "r")
rt_star = False #デフォルト値
for i,line in enumerate(f.readlines()):
    line = line.split('!')[0] #!によるcomment outの手前だけを読む
    if 'rt_star' in line:
        if '.true.' in line.lower(): #小文字にしてから判定
            rt_star = True
            print ("rt_star = .true.   -->  unit for particle_birth_epoch is physical time")
        else:
            rt_star = False
            print ("rt_star = .false.   -->  unit for particle_birth_epoch is confromal time")
f.close()

# #テーブルからcode timeとaexpの対応を取得
# #注意: シミュレーションの宇宙論パラメータとtableを作成する際の宇宙論パラメータが異なると結構 (factor ~ 2くらい) 値がずれることがあるので注意
# friedman_file=os.path.expanduser(DWARFDIR)+'/python/lib/friedman_ramses/table.dat'
# f = open(friedman_file, "r")
# aexp_frd=[]
# time_frd=[]
# for i,line in enumerate(f.readlines()):
#     aexp_frd.append(float(line.split()[0]))
#     if rt_star:
#         time_frd.append(float(line.split()[3])) #4番目のcolumn (proper time)
#     else:
#         time_frd.append(float(line.split()[2])) #3番目のcolumn (time)
# f.close()
friedman_file=None

#get_scale_factor_table.pyの関数を呼び出してcode timeとaexpの対応を取得
#注意: シミュレーションの宇宙論パラメータとtableを作成する際の宇宙論パラメータが異なると結構 (factor ~ 2くらい) 値がずれることがあるので注意
aexp_frd,hexp_frd,tau_frd,t_frd = get_scale_factor_table(omega_m,omega_l,hubble_constant*100)

if rt_star:
    time_frd = t_frd #look-back time
else:
    time_frd = tau_frd #conformal time

#過去から現在への順に並んだnumpy arrayに変換
aexp_frd = np.array(aexp_frd[::-1])
time_frd = np.array(time_frd[::-1])

#線形補間でtimeからredshiftを計算
def codetime_to_redshift(t):
    if t < time_frd[0] or time_frd[-1] < t:
        print ("time is not within the range given by the table {:}\nt = {:}, tmin = {:}, tmax = {:}\nexiting...".format(
            friedman_file, t, time_frd[0], time_frd[-1]))
        sys.exit()

    i0 = np.where(time_frd < t)[0][-1] #tを超えない最大のindex
    i1 = i0+1
    aexp = (t - time_frd[i0]) /  (time_frd[i1] - time_frd[i0]) * (aexp_frd[i1] - aexp_frd[i0]) \
        + aexp_frd[i0]
    z_redshift = 1./aexp - 1.
    return z_redshift


#線形補間でredshiftからtUniv[yr]を計算
def redshift_to_tUniv(zred):
    cgs_yr = 3.1556926e7          # 1年 (in s)
    cgs_pc = 3.08568e18           # pc (in cm)
    H0 = hubble_constant*1e2*1e5/(1e6*cgs_pc) #hubble constant H0 [km/s Mpc-1]

    aexp = 1. / (zred + 1.)
    if aexp < aexp_frd[0] or aexp_frd[-1] < aexp:
        print ("aexp is not within the range given by the table {:}\naexp = {:}, aexpmin = {:}, aexpmax = {:}\nexiting...".format(
            friedman_file, aexp, aexp_frd[0], aexp_frd[-1]))
        sys.exit()

    i0 = np.where(aexp_frd < aexp)[0][-1] #aexpを超えない最大のindex
    i1 = i0+1
    print (aexp, aexp_frd[i0], aexp_frd[i1])
    time = (aexp - aexp_frd[i0]) /  (aexp_frd[i1] - aexp_frd[i0]) * (time_frd[i1] - time_frd[i0]) \
        + time_frd[i0]
    tUniv0 = time_frd[0] * 1./H0 / cgs_yr #tableの最初の時刻を0とする (z~1e4)
    tUniv = time * 1./H0 / cgs_yr - tUniv0
    print ("t_Univ = {:.3e} yr at zred =  {:.3e}, t_Univ0 = {:.3e} yr at present".format(tUniv, zred, -tUniv0))    
    return tUniv

tUniv=redshift_to_tUniv(current_redshift)

####################################
# ハローカタログの読み込み
####################################

#Rockstar -> FOF -> HOPの順で読み込めるか調べる
#どれも存在しなければ、hc_found = False

#カタログ名
hc_name_rock = "../halo_catalogs_rock/catalog/catalog.0.h5"
hc_name_fof = "../halo_catalogs_fof/catalog/catalog.0.h5"
hc_name_hop = "../halo_catalogs_hop/catalog/catalog.0.h5"

#ハローカタログを探査
if os.path.isfile(hc_name_rock):     #Rockstarハローカタログが存在するなら
    hds_rock = yt.load(hc_name_rock)
    hc = HaloCatalog(halos_ds=hds_rock,output_dir="../halo_catalogs_rock/catalog2")
    hc_found = True
elif os.path.isfile(hc_name_fof):     #FOFハローカタログが存在するなら
    hds_fof = yt.load(hc_name_fof)
    hc = HaloCatalog(halos_ds=hds_fof,output_dir="../halo_catalogs_fof/catalog2")
    hc_found = True
elif os.path.isfile(hc_name_hop):     #HOPハローカタログが存在するなら
    hds_hop = yt.load(hc_name_hop)
    hc = HaloCatalog(halos_ds=hds_hop,output_dir="../halo_catalogs_hop/catalog2")
    hc_found = True
else:                                 #ハローカタログが存在しないなら
    hc_found = False

#ハローカタログが存在すれば読み込み
if hc_found:
    hc.load()
    hc.create()

    #halo mass
    hc_ad = hc.halos_ds.all_data()
    print ("# of halos (read from catalog):",  len(hc_ad['particle_mass']))    
    print ("halo masses (read from catalog):", np.sort(hc_ad['particle_mass'].in_units('Msun'))[::-1])    


#object_typeがhaloなのにハローカタログが存在しない場合は修了
#if object_type == "halo" and not hc_found:
if object_type in ["halo","dmax","disk"] and not hc_found:    
    print ("object_type = halo, but no halo catalog found\nexiting...")
    sys.exit()



######################################################
######################################################
#            各プロットを作成する関数                   #
######################################################
######################################################

#########################################
# プロットオブジェクトを作成して返す関数
#########################################
def generate_plot_object(var,axis,center=None,data_source=None,width_str=None,title=None,timestamp=True,velocity=True,halos=True,star_marker='cross',file_info=None,subtract_bulk_vel=True,
                         vmin=None, vmax=None):
    if data_source is None:
        data_source = ds #全体のdata source

    #bulk velocityの差っ引き
    if velocity:
        if subtract_bulk_vel:
            #オプションで与える場合
            if args.bulk_vel is not None: 
                bulk_vel = yt.YTArray([float(x) for x in args.bulk_vel.split()], "km/s")
                print ("subtract optional bulk_vel:", bulk_vel.in_units("km/s"))
            #自動で求める場合
            else:
                bulk_vel = data_source.quantities.bulk_velocity()
                #print ("(KS DEBUG) bulk_vel:", bulk_vel, type(bulk_vel))
                print ("subtract region's bulk_vel:", bulk_vel.in_units("km/s"))
        else:
            bulk_vel = yt.YTArray([0,0,0], "km/s")
        data_source.set_field_parameter("bulk_velocity", bulk_vel)


    #set slice plane from axis
    if axis == "z":
        plane = "xy"
    elif axis == "x":
        plane = "yz"
    elif axis == "y":
        # plane = "zx"
        #座標をflipさせてxz面でプロット
        plane = "xz"
        ds.coordinates.x_axis[1] = 0                   #方向1 (<-> y に垂直な方向)のプロットにおけるx軸を0番目の座標 (<-> x軸)に指定
        ds.coordinates.y_axis[1] = 2                   #方向1 (<-> y に垂直な方向)のプロットにおけるy軸を2番目の座標 (<-> z軸)に指定
        ds.coordinates.image_axis_name[1]=['x','z']    #方向1 (<-> y に垂直な方向)のプロットにおける座標のラベルを('x','z') に指定

    #変数名 -> ytの変数名
    if var in ["nH","nHproj","pStar"] :
        var_yt = "H_nuclei_density"
    elif var in ["T","Tproj"]:
        var_yt = "my_temperature"
    elif var == "Mach":
        var_yt = "Mach"
    elif var == "metal":
        #var_yt = "Zmetal_in_Zsun"
        var_yt = "Zmetal_in_Zsun_floor"    #zero-metal gasの場合にlabelが壊れてエラーになりそれ以降のプロットが作れなくなる現象が発生したのでフロア付きを利用
    elif var == "yHI":
        var_yt = "yHI"
    elif var == "yHII":
        var_yt = "yHII"
    elif var == "yH2":
        var_yt = "yH2"
    elif var == "Sigma":
        var_yt = "density"
    elif var == "sStar":
        var_yt = "star_cic"
    elif var == "sDM":
        var_yt = "DM_cic"        
    elif var in ["pDM","DMhalo"]:
        var_yt = ('DM', 'particle_mass') #単にparticle_massだとうまくいかない
    elif var in ["phi","phiproj"]:
        var_yt = "potential"
    elif var == 'J21':
        var_yt = 'J21'

    if  (data_source is None) or (width_str is None) or (center is None):
        print ("Currently, all of data_source, width, and center should be given to generate_plot().\nexiting...")
        sys.exit()

    if star_marker not in ['cross','dot', None]:
        print ("star_marker option for generate_plot should be either 'cross' or 'dot'\nset it to None")
        star_marker = None

    #widthを文字列から値に変換
    if width_str == 'whole':
        width = boxsize_in_kpc
    else:
        if width_str[0]=='0':                          #0から始まる場合は小数
            n_zeros = width_str.count('0')            #003 -> 2
            width = ds.quan(float(width_str)*10**(-n_zeros), "kpc") #003 -> 0.03            
        else:                                         #通常の場合                             
            width = ds.quan(float(width_str), "kpc")  #30 -> 30

    #よくわからないが、widthをcode_unitにしないとバグが発生する場合があるかも
    width = width.in_units('code_length')

    #axes_unitの設定をボックスサイズに従って変更
    if width <  ds.quan(1e0, "kpc"):     #プロット領域が1kpc以下の時
        axes_unit = 'pc'
    elif width <  ds.quan(1e3, "kpc"):   #プロット領域が1Mpc以下の時
        axes_unit = 'kpc'
    else:                                #プロット領域が1Mpc以上の時
        axes_unit = 'Mpc'
        
    #プロット作成
    #------------------- disk以外の場合 ----------------#
    if object_type != "disk":
        if var in ["nH","T","Mach","metal","yHI","yHII","yH2","phi","J21"]:
            p=yt.SlicePlot(ds, axis, var_yt,center=center,data_source=data_source,width=width,axes_unit=axes_unit)
        elif var in ["Sigma","sStar","sDM"]:
            #data_source.max_level = max(0,data_source.max_level - 1) #** ERROR ** this leads to strange cic plots
            p=yt.ProjectionPlot(ds, axis, var_yt,center=center,data_source=data_source,width=width,axes_unit=axes_unit)
            p.set_unit(var_yt, 'Msun/pc**2')    
        elif var in ["nHproj","Tproj","pStar","phiproj"]:
            #weighted projection
            p=yt.ProjectionPlot(ds, axis, var_yt,center=center,data_source=data_source,weight_field='density',width=width,axes_unit=axes_unit)
        elif var in ["pDM","DMhalo"]:
            if axis == "z":
                p = yt.ParticlePlot(ds, ('DM','particle_position_x'), ('DM','particle_position_y'),('DM','particle_mass'),
                                    center=center,data_source=data_source,width=width,axes_unit=axes_unit) #xy平面
            elif axis == "x":
                p = yt.ParticlePlot(ds, ('DM','particle_position_y'), ('DM','particle_position_z'),('DM','particle_mass'),
                                    center=center,data_source=data_source,width=width,axes_unit=axes_unit) #yz平面
                p.set_buff_size(320)

            elif axis == "y":
                p = yt.ParticlePlot(ds, ('DM','particle_position_x'), ('DM','particle_position_z'),('DM','particle_mass'),
                                    center=center,data_source=data_source,width=width,axes_unit=axes_unit) #xz平面
            #particle plot resolution
            if args.buff_size is not None:
                p.set_buff_size(args.buff_size)
            else:
                p.set_buff_size(320)       #yt default value = 800 is too large
            p.set_unit(('DM','particle_mass'), 'Msun')    

    #------------------- diskの場合 ----------------#
    else: #object_type == "disk"
        #define axis
        yax0 = np.array([0.,1.,0,])                                 # use original y-axis vector to obtain x-axis
        zax = polar_axis                                            # z-axis vector
        xax = np.cross(yax0,zax)/np.linalg.norm(np.cross(yax0,zax)) # x-axis vector
        yax = np.cross(zax,xax)/np.linalg.norm(np.cross(zax,xax))   # y-axis vector
        if axis == "z":           #xy-plane
            normal_vector = zax            
            north_vector = yax        
        elif axis == "y":           #xz-plane
            normal_vector = yax            
            north_vector = zax        
        elif axis == "x":           #yz-plane
            normal_vector = xax            
            north_vector = zax        
        print ("axis, normal_vector, north_vector: ", axis, normal_vector, north_vector)

        #call plot function
        if var in ["nH","T","Mach","metal","yHI","yHII","yH2","phi"]:
            p=yt.OffAxisSlicePlot(ds,normal_vector,var_yt,center=center,data_source=data_source,width=width,axes_unit=axes_unit,north_vector=north_vector)
        elif var in ["Sigma","sStar","sDM"]:
            p=yt.OffAxisProjectionPlot(ds,normal_vector,var_yt,center=center,data_source=data_source,width=width,axes_unit=axes_unit,north_vector=north_vector)
            p.set_unit(var_yt, 'Msun/pc**2')    
        elif var in ["nHproj","Tproj","phiproj"]:
            #weighted projection
            p=yt.OffAxisProjectionPlot(ds,normal_vector,var_yt,center=center,data_source=data_source,weight_field='density',width=width,axes_unit=axes_unit,north_vector=north_vector)
        elif var in ["pStar","pDM","DMhalo"]:
            print ("var = {:} is not supported for object_type = disk. Skip plot...".format(var))
            return


        # elif var in []:
        #     if axis == "z":
        #         p = yt.ParticlePlot(ds, ('DM','particle_position_x'), ('DM','particle_position_y'),('DM','particle_mass'),
        #                             center=center,data_source=data_source,width=width,axes_unit=axes_unit) #xy平面
        #     elif axis == "x":
        #         p = yt.ParticlePlot(ds, ('DM','particle_position_y'), ('DM','particle_position_z'),('DM','particle_mass'),
        #                             center=center,data_source=data_source,width=width,axes_unit=axes_unit) #yz平面
        #         p.set_buff_size(320)

        #     elif axis == "y":
        #         p = yt.ParticlePlot(ds, ('DM','particle_position_x'), ('DM','particle_position_z'),('DM','particle_mass'),
        #                             center=center,data_source=data_source,width=width,axes_unit=axes_unit) #xz平面
        #     #particle plot resolution
        #     if args.buff_size is not None:
        #         p.set_buff_size(args.buff_size)
        #     else:
        #         p.set_buff_size(320)       #yt default value = 800 is too large
        #     p.set_unit(('DM','particle_mass'), 'Msun')    



    #annotation等
    # if velocity: p.annotate_velocity(plot_args={"headwidth": 4}, factor=32) #velocity annotation

    if velocity: 
        if not (object_type == "disk" and var in ["Sigma","sStar","sDM", "nHproj","Tproj","phiproj"]):
            p.annotate_velocity(plot_args={"headwidth": 4}, factor=32) #velocity annotation

    
    if timestamp:                                                           #timestamp annotation
        #p.annotate_timestamp(time=False,redshift=True)
        p.annotate_timestamp(time=False,redshift=True,
                             text_args={'color':'black','bbox':dict(boxstyle='round', facecolor='white', alpha=0.8)})
    if halos:                                                               #halo annotation
        if not hc_found:
            print ("Halo catalog not found\nskip halo annotation")
        else:
            if var == "DMhalo":
                #p.annotate_halos(hc,circle_args={"edgecolor":"black","fill":False})
                #p.annotate_halos(hc,width=width,circle_args={"edgecolor":"black","fill":False})
                #なぜかwidthをそのまま渡すとエラーになるので値に一度変換 (ytにありがちな現象)
                width_Mpc = width.in_units('Mpc').to_value() 
                p.annotate_halos(hc,width=(width_Mpc,'Mpc'),circle_args={"edgecolor":"black","fill":False})
            else:
                #p.annotate_halos(hc)
                #なぜかwidthをそのまま渡すとエラーになるので値に一度変換 (ytにありがちな現象)
                width_Mpc = width.in_units('Mpc').to_value() 
                p.annotate_halos(hc,width=(width_Mpc,'Mpc'))
                
    if title != None:                                                      #title annotation
        #p.annotate_title(title)
        #p.annotate_title(title)
        p.annotate_text((0.05, 0.96), title, coord_system='figure',
                text_args={'color':'black', 'size':12})        
    p.annotate_scale()

    if star_marker == 'cross':                                                               #particle annotation
        p.annotate_particles(width,ptype='star', p_size=200.0,marker='x',col='r',)
        p.annotate_particles(width,ptype='POPIII', p_size=200.0,marker='x', col='w',)
        p.annotate_particles(width,ptype='supernova', p_size=200.0,marker='x',col='m',)
        p.annotate_particles(width,ptype='dead', p_size=200.0,marker='x',col='b',)
        p.annotate_particles(width,ptype='BH', p_size=200.0,marker='x',col='k',)
    elif star_marker == 'dot':                                                              #particle annotation
        #p.annotate_particles(width,ptype='star', p_size=20.0,marker='.',col='r',)
        p.annotate_particles(width,ptype='star', p_size=20.0,marker='.',col='r',)
        p.annotate_particles(width,ptype='POPIII', p_size=20.0,marker='.', col='w',)
        p.annotate_particles(width,ptype='supernova', p_size=20.0,marker='.',col='m',)
        p.annotate_particles(width,ptype='dead', p_size=20.0,marker='.',col='b',)
        p.annotate_particles(width,ptype='BH', p_size=200.0,marker='x',col='k',)
        

    if title is not None:                                                      #file info
        p.annotate_text((0.01, 0.01), file_info, coord_system='figure',
                text_args={'color':'black', 'size':10})

    #color barの設定
    if var == "metal":
        p.set_log(var_yt, True)
        p.set_zlim(var_yt, 1e-6, 1e0)
        fmt_func = lambda x,pos: "{:}".format(-int(np.round(x))) #e.g., Z = 1e-4 Zsun  ->  4
        fmt = matplotlib.ticker.FuncFormatter(fmt_func)
        p.annotate_contour(var_yt, ncont=6, clim=(1e-5,1e0), label=True,text_args={'fmt':fmt},
                           plot_args={"colors": "w","linewidths": 1})
        # p.annotate_contour(var_yt, ncont=6, clim=(1e-5,1e0), label=True,
        #             plot_args={"colors": "r","linewidths": 1})        
        # p.annotate_contour(var_yt, ncont=1, clim=(1e-5,1e-5), label=True,text_args={'fmt':fmt},                           
        #             plot_args={"colors": "r","linewidths": 2})
        # p.annotate_contour(var_yt, ncont=1, clim=(1e-5,1e-5), label=True,text_args={'fmt':fmt},                           
        #             plot_args={"colors": "w","linewidths": 1})        

    #color barの上限・下限を設定 (Noneは値を変えない)
    if vmin is not None and vmax is not None:
        p.set_zlim(var_yt, vmin, vmax)
    elif (vmin is not None and vmax is None) or  (vmin is None and vmax is not None):
        print ("both or none of vmin and vmax should be specified, but not either one in the current version of the code (YT's specification??)\nexiting...")
        sys.exit()

    #color mapの設定
    if var_yt == 'my_temperature':
        p.set_cmap(field=var_yt, cmap = 'hot')
    elif var_yt == 'Zmetal_in_Zsun':
        p.set_cmap(field=var_yt, cmap = 'dusk_r')
    elif var_yt in ['H_nuclei_density', ('DM', 'particle_mass')]:
        p.set_cmap(field=var_yt, cmap = 'arbre')
    elif var_yt in ["yHI","yHII","yH2"]:
        p.set_cmap(field=var_yt, cmap = 'kelp')
    elif var_yt in ["density", "star_cic", "DM_cic"]:
        p.set_cmap(field=var_yt, cmap = 'jet')
    elif var_yt in ["J21"]:
        p.set_cmap(field=var_yt, cmap = 'plasma')
    else:
        #p.set_cmap(field=var_yt, cmap = 'algae')
        p.set_cmap(field=var_yt, cmap = 'arbre')

    #重力ポテンシャルのプロットはリニアスケールで
    if var in ["phi","phiproj"]:
        p.set_log(var_yt, False)        

    return p

#########################################
# プロットオブジェクトからプロットを作成して保存
#########################################
def generate_plot(var,axis,center=None,data_source=None,width_str=None,title=None,timestamp=True,velocity=True,halos=True,star_marker='dot',file_info=None,subtract_bulk_vel=True,
                  vmin=None, vmax=None):

    print ("\ngenerate plot: var = {:}, width = {:}, axis = {:}".format(var, width_str, axis))

    #set slice plane from axis
    if axis == "z":
        plane = "xy"
    elif axis == "x":
        plane = "yz"
    elif axis == "y":
        # plane = "zx"
        #座標をflipさせてxz面でプロット
        plane = "xz"
    elif axis == "xyz":
        plane = "xyz" #3 penel plot for 3 planes

    #set output file name
    if width_str == 'whole':
        outname = "%s_%s_%s_whole_s%s.png"%(outname_prefix,var,plane,step)
    else:
        outname = "%s_%s_%s_%skpc_s%s.png"%(outname_prefix,var,plane,width_str,step)

   
    if axis != "xyz":    #通常プロット
        #get plot object
        p = generate_plot_object(var,axis,center,data_source,width_str,title,timestamp,velocity,halos,star_marker,file_info,subtract_bulk_vel,vmin=vmin,vmax=vmax)
        #save figure
        p.set_figure_size(5)
        p.save(outname)

        
    else:                #3 panel プロット (cf. https://stackoverflow.com/questions/32954812/adding-subplots-within-yt-project-plot)
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import AxesGrid

        fig = plt.figure()
        grid = AxesGrid(fig, (0.05,0.1,0.8,0.8),
                nrows_ncols = (2, 2),
                axes_pad = 0.001 ,
                label_mode = "L",
                share_all = True,
                cbar_location="right",
                cbar_mode="single",
                cbar_size="3%",
                cbar_pad="0%") 

        #1st panel (xy plane)
        p1 = generate_plot_object(var,"z",center=center,data_source=data_source,width_str=width_str,
                                 title=None,timestamp=False,velocity=velocity,halos=halos,star_marker=star_marker,
                                 file_info=file_info,subtract_bulk_vel=subtract_bulk_vel,vmin=vmin,vmax=vmax)


        #2nd panel (xz plane)
        p2 = generate_plot_object(var,"y",center=center,data_source=data_source,width_str=width_str,
                                 title=None,timestamp=False,velocity=velocity,halos=halos,star_marker=star_marker,
                                 file_info=file_info,subtract_bulk_vel=subtract_bulk_vel,vmin=vmin,vmax=vmax)

        #3rd panel (yz plane)
        p3 = generate_plot_object(var,"x",center=center,data_source=data_source,width_str=width_str,
                                 title=None,timestamp=False,velocity=velocity,halos=halos,star_marker=star_marker,
                                 file_info=file_info,subtract_bulk_vel=subtract_bulk_vel,vmin=vmin,vmax=vmax)


        #plotされている量
        pvar = list(p1.plots.keys())[0] 

        #vmin/vmaxが与えられていない場合に自動設定
        if vmin is None or vmax is None:
            #image中の値の最小・最大 (frbで2D map中の各セルの値が取得可能)
            #描画領域がデータより大きいときにできる空白にはゼロが入ってしまうので取り除いてmin/maxをとる
            #particle plotの場合はデータが無い領域にはnanが入る
            #有効なデータ無しの場合はdefault値?
            p1frb_nonzero= p1.frb[pvar][p1.frb[pvar]!=0] #non-zeroに限定
            p2frb_nonzero= p2.frb[pvar][p2.frb[pvar]!=0] #non-zeroに限定
            p3frb_nonzero= p3.frb[pvar][p3.frb[pvar]!=0] #non-zeroに限定

            #全てのパネルに有効データが少なくとも一つ以上ある場合
            if len(p1frb_nonzero) > 0 and  len(p2frb_nonzero) > 0  or len(p3frb_nonzero) > 0:
                #min
                vmin = np.nanmin(p1frb_nonzero)              #nanmin: nanを除いたmin
                vmin = min(vmin,np.nanmin(p2frb_nonzero))
                vmin = min(vmin,np.nanmin(p3frb_nonzero))
                #max
                vmax = np.nanmax(p1frb_nonzero)
                vmax = max(vmax,np.nanmax(p2frb_nonzero))
                vmax = max(vmax,np.nanmax(p3frb_nonzero))
                print ("vmin/vmax of 3 panels: ", pvar, vmin, vmax)


                #重力ポテンシャルプロットだけリニアスケールなので特別
                if var not in ["phi","phiproj"]:
                
                    #vminが小さすぎてプロットが見にくくなることがあったので対応 (場合によっては余計かも)
                    if vmin < 1e-10:
                        print ('** WARNING ** vmin is very large. Change it to 1e-10 for visibility of the plot. Manually setting vmin can skip this process.')
                        vmin = 1e-10

                    #vmin/vmaxが近すぎると目盛りが表示されなくなるので、最低でも一桁の幅を取るようにする
                    if vmax/vmin < 1e1:
                        fact = np.sqrt(1e1 / (vmax/vmin))
                        print ('ratio of vmax and vmin is {:} (< 10)\nensure at least one order difference by setting new vmin/vmax = {:}  {:}'.format(vmax/vmin, vmax*fact, vmin/fact))
                        vmin = vmin/fact
                        vmax = vmax*fact


            #有効データが存在しないパネルがある場合はデフォルト値
            else:
                vmin = 1
                vmax = 10
                print ("use dummy vmin/vmax because automatic determination fails: ", pvar, vmin, vmax)

        #各パネルをAxesGrid上に配置
        #1st panel
        p1.set_zlim(pvar,vmin,vmax)
        plot = p1.plots[pvar]
        plot.figure = fig
        plot.axes = grid[0].axes
        plot.cax = grid.cbar_axes[0]
        p1._setup_plots() #redraws the plot

        #2nd panel
        p2.set_zlim(pvar,vmin,vmax)
        plot = p2.plots[pvar]
        plot.figure = fig
        plot.axes = grid[2].axes
        plot.cax = grid.cbar_axes[2]
        p2._setup_plots() #redraws the plot

        #3rd panel
        p3.set_zlim(pvar,vmin,vmax)
        plot = p3.plots[pvar]
        plot.figure = fig
        plot.axes = grid[3].axes
        plot.cax = grid.cbar_axes[3]
        p3._setup_plots() #redraws the plot

        #重力ポテンシャルプロットだけリニアスケールなので特別
        if var not in ["phi","phiproj"]:

            #plotのdynamic rangeが小さい場合はcolor barにminor ticksを追加
            if vmax/vmin < 1e5:
                grid.cbar_axes[0].yaxis.set_minor_locator(matplotlib.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10))



        #---------------------------------------
        #余白に情報を書き込み
        #---------------------------------------

        #widthを文字列から値に変換
        if width_str == 'whole':
            width = boxsize_in_kpc
        else:
            if width_str[0]=='0':                          #0から始まる場合は小数
                n_zeros = width_str.count('0')            #003 -> 2
                width = ds.quan(float(width_str)*10**(-n_zeros), "kpc") #003 -> 0.03            
            else:                                         #通常の場合                             
                width = ds.quan(float(width_str), "kpc")  #30 -> 30
        #単位を調整
        if width <  ds.quan(1e0, "kpc"):     #プロット領域が1kpc以下の時
            width = width.in_units('pc')
        elif width <  ds.quan(1e3, "kpc"):   #プロット領域が1Mpc以下の時
            width = width.in_units('kpc')
        else:                                #プロット領域が1Mpc以上の時
            width = width.in_units('Mpc')


        #プロット対象
        if outname_prefix == "sim":
            txt="simulation box"
        elif outname_prefix[:4] == "halo":
            txt= r"halo {:}:".format(i_obj) + "\n" + \
                r"$M_{{vir}} = {:.2g}\ M_\odot, r_\mathrm{{vir}} = {:3.1f}\ \mathrm{{kpc}}$".format(float(mass_obj),float(rvir_obj))
        elif outname_prefix[:4] == "dmax":
            txt= r"halo {:} at nHmax = {:.2e}:".format(i_obj,nHmax) + "\n" + \
                r"$M_{{vir}} = {:.2g}\ M_\odot, r_\mathrm{{vir}} = {:3.1f}\ \mathrm{{kpc}}$".format(float(mass_obj),float(rvir_obj))
        elif outname_prefix[:4] == "disk":
            txt= r"halo {:} with disk axis:".format(i_obj) + "\n" + \
            r"$M_{{vir}} = {:.2g}\ M_\odot, r_\mathrm{{vir}} = {:3.1f}\ \mathrm{{kpc}}$".format(float(mass_obj),float(rvir_obj)) \
            + "\n" + "polar axis = ({:.3f}, {:.3f}, {:.3f})".format(polar_axis[0],polar_axis[1],polar_axis[2])
        elif outname_prefix[:3] == "idp":
            txt= r"idp {:}, type {:}:".format(i_obj,star_type) + "\n" + \
                r"$ M_\mathrm{{star}} = {:4.1f}\ M_\odot, z_\mathrm{{form}} = {:6.2f}$".format(float(mass_obj),birth_redshift_obj)
        else:
            txt="something wrong?"    
        fig.text(0.45,0.84,txt, size=12)

        #step数/プロット幅/プロット中心/redshift
        fig.text(0.45,0.8,"#{:}".format(step), size=12)  #step数
        fig.text(0.45,0.77,"width={:}".format(width), size=12) #プロット幅
        fig.text(0.45,0.74,"cen=[{:.7f}, {:.7f}, {:.7f}]".format(float(center[0]),float(center[1]),float(center[2])), size=12) #プロットの中心
        fig.text(0.45,0.71,"z={:.9f}, tUniv={:.6e} yr".format(current_redshift,tUniv),transform=fig.transFigure, size=12) #redshift
        #fig.text(0.45,0.71,"z={:.9f}".format(current_redshift),transform=fig.transFigure, size=12) #redshift

        #file info
        fig.text(0.01, 0.01, file_info, transform=fig.transFigure, size=10) 

        #plot var/size
        fig.text(0.9, 0.98, '{:}/{:}'.format(var,width_str), transform=fig.transFigure, size=10) 


        #フレームの速度
        if velocity and subtract_bulk_vel:
            bulk_vel = data_source.quantities.bulk_velocity().in_units("km/s")
            fig.text(0.45,0.68,"bulk_vel=[{:.2f}, {:.2f}, {:.2f}] km/s".format(
                float(bulk_vel[0]),float(bulk_vel[1]),float(bulk_vel[2])), size=12)

        #particle情報
        # fig.text(0.45,0.65,"particle # (DM, Pop2, Pop3, SN, dead, BH)", size=12) 
        # fig.text(0.45,0.63,'= {:}, {:}, {:}, {:}, {:}, {:}'.format(N_DM, N_pop2, N_pop3,N_sn,N_dead,N_BH), size=12) 
        # fig.text(0.45,0.60,'particle mass [Msun] (DM, Pop2, Pop3_tot)', size=12)
        # fig.text(0.45,0.58,'= {:.2e},  {:.2e},  {:.2e}'.format(M_DM.to_value() ,M_pop2.to_value() ,(M_pop3+M_sn+M_dead+M_BH).to_value()), size=12)
        fig.text(0.45,0.65,"particle # (DM, Pop2, Pop3, BH, SFC, PSC)", size=12) 
        fig.text(0.45,0.63,'= {:}, {:}, {:}, {:}, {:}, {:}'.format(N_DM, N_pop2,N_pop3,N_BH,N_SFC,N_PSC), size=12) 
        fig.text(0.45,0.60,'particle mass [Msun] (DM, Pop2, Pop3)', size=12)
        fig.text(0.45,0.58,'= {:.2e},  {:.2e},  {:.2e}'.format(M_DM.to_value() ,M_pop2.to_value() ,M_pop3.to_value()), size=12) 
        
        

        #save plot
        print ("save {:}".format(outname))
        plt.savefig(outname, bbox_inches='tight')


        

    
#########################################
# phase diagramを作成する関数                  
#########################################
# 描画されるbinに含まれるmassがゼロの時にerrorが発生する問題あり (KS TODO)
def generate_phase_plot(var,data_source=None,width_str='whole',file_info=None,xmin=None,xmax=None,ymin=None,ymax=None):

    print ("\ngenerate phase plot: var = {:}, width = {:}".format(var, width_str))

    if data_source is None:
        data_source = ds.all_data()

    #widthを文字列から値に変換
    if width_str == 'whole':
        width = boxsize_in_kpc
    else:
        if width_str[0]=='0':                          #0から始まる場合は小数
            n_zeros = width_str.count('0')            #003 -> 2
            width = ds.quan(float(width_str)*10**(-n_zeros), "kpc") #003 -> 0.03
        else:                                         #通常の場合                             
            width = ds.quan(float(width_str), "kpc")  #30 -> 30

    #よくわからないが、widthをcode_unitにしないとバグが発生する場合があるかも
    width = width.in_units('code_length')

    if var == 'nT':
        varX_yt = 'H_nuclei_density'
        varY_yt = 'my_temperature'        
    elif var == 'MachT':
        varX_yt = 'Mach'
        varY_yt = 'my_temperature'
    elif var == 'nmetal':
        varX_yt = 'H_nuclei_density'
        varY_yt = 'Zmetal_in_Zsun_floor'
    elif var == 'nyH2':
        varX_yt = 'H_nuclei_density'
        varY_yt = 'yH2_floor'        
    elif var == 'nyHII':
        varX_yt = 'H_nuclei_density'
        varY_yt = 'yHII'        
    elif var == 'metalyH2':
        varX_yt = 'Zmetal_in_Zsun_floor'
        varY_yt = 'yH2_floor'        


    varX_min =  data_source.min(varX_yt)
    varX_max =  data_source.max(varX_yt)
    varY_min =  data_source.min(varY_yt)
    varY_max =  data_source.max(varY_yt)
      
    if rank == 0:
        print ("(phase plot) x var = {:}: min = {:}, max = {:}".format(varX_yt,varX_min,varX_max))
        print ("(phase plot) y var = {:}: min = {:}, max = {:}".format(varY_yt,varY_min,varY_max))

    #logプロットが壊れないようにfloorを課す
    varX_plotmin = max(varX_min,1e-11)
    varY_plotmin = max(varY_min,1e-11)
    varX_plotmax = max(varX_max,1e1*varX_plotmin)
    varY_plotmax = max(varY_max,1e1*varY_plotmin)
    

    #プロット領域にちょっと余裕をもたせる
    varX_plotmin = 10**np.floor(np.log10(varX_plotmin))
    varX_plotmax = 10**np.floor(np.log10(varX_plotmax)+1)
    varY_plotmin = 10**np.floor(np.log10(varY_plotmin))
    varY_plotmax = 10**np.floor(np.log10(varY_plotmax)+1)

    #vmin/vmaxが与えられていたらそっちを利用
    #プロット範囲内にデータが無い場合はreturn
    if xmin is not None:
        if xmin > varX_max:
            print ('xmin (= {:}) > varX_max (= {:})  skip this plot'.format(xmin, varX_max))
            return
        varX_plotmin = xmin
    if xmax is not None:
        if xmax < varX_min:
            print ('xmax (= {:}) < varX_min (= {:})  skip this plot'.format(xmax, varX_min))
            return
        varX_plotmax = xmax
    if ymin is not None:
        if ymin > varY_max:
            print ('ymin (= {:}) > varY_max (= {:})  skip this plot'.format(ymin, varY_max))
            return
        varY_plotmin = ymin
    if ymax is not None:
        if ymax < varY_min:
            print ('ymax (= {:}) < varY_min (= {:})  skip this plot'.format(ymax, varY_min))
            return
        varY_plotmax = ymax    
    

    units = dict(cell_mass='Msun')
    extrema = {varX_yt:(varX_plotmin,varX_plotmax), varY_yt:(varY_plotmin,varY_plotmax)}
    profile = yt.create_profile(data_source, [varX_yt,varY_yt],
                            n_bins=[128, 128], fields=['cell_mass'],
                            weight_field=None, units=units, extrema=extrema)

    p = yt.PhasePlot.from_profile(profile)
    p.set_figure_size(5)    

    if width_str == 'whole':
        outname = "%s_%s_whole_s%s.png"%(outname_prefix,var,step)
    else:
        outname = "%s_%s_%skpc_s%s.png"%(outname_prefix,var,width_str,step)


    #axesを取得して、無理矢理気味にtextを自由に追加し、図を保存し直す（一度図を作った後でないと、axesが取得できない）
    p.save(outname)
    f=p.profile.items()[0][0]

    #axes/caxを取得してpositionを手動で設定
    axes = p.plots[f].axes
    axes.set_position([0.2,0.2,0.6,0.5])
    cax = p.plots[f].cax
    cax.set_position([0.8,0.2,0.02,0.5])

    #x軸のサブ目盛りを1桁ずつ書く
    axes.xaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10., numticks=20))
    axes.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


    #star formationのthreshold curveを描画
    if args.nT_nth is not None and var == 'nT':
        if rank == 0:
            print ('Pop II star formation threshold density  = {:.2e} ((1+z)/10)^2 * (T2/100K) cm-3'.format(args.nT_nth))
        nth_T100 = args.nT_nth * ((1.+current_redshift)/1e1)**2 #nth at T=100K at given redshift
        nth0 = nth_T100 * (varY_plotmin/1e2) #nth is proportional to T
        nth1 = nth_T100 * (varY_plotmax/1e2) #nth is proportional to T    
        axes.plot([nth0,nth1],[varY_plotmin,varY_plotmax],'r--')

    
    #---------------------------------------
    #余白に情報を書き込み
    #---------------------------------------
    fig=axes.figure #axesの所属しているfigureを取得

    #widthを文字列から値に変換
    if width_str == 'whole':
        width = boxsize_in_kpc
    else:
        if width_str[0]=='0':                          #0から始まる場合は小数
            n_zeros = width_str.count('0')            #003 -> 2
            width = ds.quan(float(width_str)*10**(-n_zeros), "kpc") #003 -> 0.03            
        else:                                         #通常の場合                             
            width = ds.quan(float(width_str), "kpc")  #30 -> 30
    #単位を調整
    if width <  ds.quan(1e0, "kpc"):     #プロット領域が1kpc以下の時
        width = width.in_units('pc')
    elif width <  ds.quan(1e3, "kpc"):   #プロット領域が1Mpc以下の時
        width = width.in_units('kpc')
    else:                                #プロット領域が1Mpc以上の時
        width = width.in_units('Mpc')


    #プロット対象
    if outname_prefix == "sim":
        txt="simulation box"
    elif outname_prefix[:4] == "halo":
        txt= r"halo {:}:".format(i_obj) + "\n" + \
            r"$M_{{vir}} = {:.2g}\ M_\odot, r_\mathrm{{vir}} = {:3.1f}\ \mathrm{{kpc}}$".format(float(mass_obj),float(rvir_obj))
    elif outname_prefix[:4] == "dmax":
        txt= r"halo {:} at nHmax = {:.2e}:".format(i_obj,nHmax) + "\n" + \
            r"$M_{{vir}} = {:.2g}\ M_\odot, r_\mathrm{{vir}} = {:3.1f}\ \mathrm{{kpc}}$".format(float(mass_obj),float(rvir_obj))
    elif outname_prefix[:3] == "idp":
        txt= r"idp {:}, type {:}:".format(i_obj,star_type) + "\n" + \
            r"$ M_\mathrm{{star}} = {:4.1f}\ M_\odot, z_\mathrm{{form}} = {:6.2f}$".format(float(mass_obj),birth_redshift_obj)
    else:
        txt="something wrong?"    
    fig.text(0.01,0.90,txt, size=12)

    #step数/プロット幅/プロット中心/redshift
    fig.text(0.01,0.85,"#{:}".format(step), size=12)  #step数
    fig.text(0.01,0.81,"width={:}".format(width), size=12) #プロット幅
    fig.text(0.01,0.77,"cen=[{:.7f}, {:.7f}, {:.7f}]".format(float(center[0]),float(center[1]),float(center[2])), size=12) #プロットの中心
    #fig.text(0.01,0.73,"z={:.8f}".format(current_redshift),transform=fig.transFigure, size=12) #redshift
    fig.text(0.01,0.73,"z={:.8f}, tUniv={:.9e} yr".format(current_redshift,tUniv),transform=fig.transFigure, size=12) #redshift

    #file info
    fig.text(0.01, 0.01, file_info, transform=fig.transFigure, size=10) 

    #particle情報
    # fig.text(0.45,0.95,"particle # (DM, Pop2, Pop3, SN, dead, BH)", size=12) 
    # fig.text(0.45,0.92,'= {:}, {:}, {:}, {:}, {:}, {:}'.format(N_DM, N_pop2, N_pop3,N_sn,N_dead,N_BH), size=12) 
    # fig.text(0.45,0.87,'particle mass [Msun] (DM, Pop2, Pop3_tot)', size=12)
    # fig.text(0.45,0.84,'= {:.2e},  {:.2e},  {:.2e}'.format(M_DM.to_value() ,M_pop2.to_value() ,(M_pop3+M_sn+M_dead+M_BH).to_value()), size=12)
    fig.text(0.45,0.95,"particle # (DM, Pop2, Pop3, BH, SFC, PSC)", size=12) 
    fig.text(0.45,0.92,'= {:}, {:}, {:}, {:}, {:}, {:}'.format(N_DM, N_pop2,N_pop3,N_BH,N_SFC,N_PSC), size=12) 
    fig.text(0.45,0.87,'particle mass [Msun] (DM, Pop2, Pop3)', size=12)
    fig.text(0.45,0.84,'= {:.2e},  {:.2e},  {:.2e}'.format(M_DM.to_value() ,M_pop2.to_value() ,M_pop3.to_value()), size=12)              



    # if width == boxsize_in_kpc:
    #     axes.text(0.6,0.02,'z = {:.4f}'.format(current_redshift),
    #               transform=axes.transAxes,fontsize=16)
    # else:
    #     axes.text(0.6,0.05,'r < {:.2g} kpc\nz = {:.4f}'.format(
    #         float(width.in_units("kpc")),current_redshift),
    #               transform=axes.transAxes,fontsize=16)gg

    #axes.set_title(title,fontsize=14,loc='right')
    p.save(outname)


#########################################
# MF plotを作成する関数                  
#########################################
def generate_MF_plot(center=None,width_str=None):
    #-----------------------------------------------------
    # halo dsからhalo mass functionを作成する関数 (yt_attic/halo_mass_function/halo_mass_function.pyを参考)
    # input:
    #  halo_ds
    #  center, width_str: boxを定義してその中でのMFを取得
    # output:
    #  hmasses: mass functionのbin in [Msun/h]
    #  ncuml: cumulative number of halo n(>M) in [Mpc^3/h^3]
    #-----------------------------------------------------
    def get_hmass_func(halos_ds, center, width_str):
        data_source = halos_ds.all_data()
        hmasses = []

        #widthを文字列から値に変換
        if width_str == 'whole':
            width = boxsize_in_kpc
        else:
            if width_str[0]=='0':                          #0から始まる場合は小数
                n_zeros = width_str.count('0')            #003 -> 2
                width = ds.quan(float(width_str)*10**(-n_zeros), "kpc") #003 -> 0.03
            else:                                         #通常の場合                             
                width = ds.quan(float(width_str), "kpc")  #30 -> 30
        print ("MF for a box with {:} on a side".format(width))

        #box内のハローのみを取得
        for i in range(len(data_source["particle_mass"])):
            if (center is None) or (width is None):
                hmasses.append(data_source["particle_mass"][i].in_units("Msun/h"))
            else:
                hpos = data_source["particle_position"][i]
                if (abs(hpos[0] - center[0]) < 0.5*width and 
                    abs(hpos[1] - center[1]) < 0.5*width and
                    abs(hpos[2] - center[2]) < 0.5*width):
                    hmasses.append(data_source["particle_mass"][i].in_units("Msun/h"))
                
        #cumulative number densityを取得
        hmasses = np.sort(hmasses)
        width_Mpccm_h = width.in_units('Mpccm/h')
        sim_volume = width_Mpccm_h**3
        n_cumulative = np.arange(len(hmasses),0,-1)
        hmasses, unique_indices = np.unique(hmasses, return_index=True)
        n_cumulative = n_cumulative[unique_indices] /sim_volume
        return hmasses, n_cumulative
    
    #シミュレーションのMFを取得してプロット
    #hmasses,n_cumulative = get_hmass_func(hc.halos_ds)
    hmasses,n_cumulative = get_hmass_func(hc.halos_ds, center, width_str)    

    if (rank == 0):
        print ("hmasses:",hmasses)
        print ("n_cumulative:",n_cumulative)

        #plot
        plt.figure() #プロットの初期化になるらしい
        #plt.loglog(hmasses, n_cumulative, label='Simulation')
        plt.loglog(hmasses, n_cumulative, label='width={:}, redshift={:.3e}'.format(width_str, current_redshift))

        #simulationデータに基づいてプロット範囲を指定
        xmin,xmax=plt.xlim()
        #plt.xlim(10**np.floor(np.log10(xmin)-1), 10**(np.floor(np.log10(xmax)+2)))
        plt.xlim(10**np.floor(np.log10(xmin)-1), 10**(np.floor(np.log10(xmax)+2)))    
        ymin,ymax=plt.ylim()
        plt.ylim(10**np.floor(np.log10(ymin)-1), 10**(np.floor(np.log10(ymax)+2)))

        #hmfモジュールによる解析的なHalo MFを取得してプロット
        from hmf import MassFunction
        import camb
        p = camb.CAMBparams()
        p.set_matter_power(kmax=100)   # sensitive to k_per_logint as well
        z = max(current_redshift,0) #hmfはz>=0しか受け付けない仕様のよう
        h_smt = MassFunction(z=z,Mmin=4, Mmax=16,transfer_params={"camb_params":p},hmf_model="SMT")
        dm = h_smt.m[1:]-h_smt.m[:-1]
        dndm_bar = (h_smt.dndm[1:]+h_smt.dndm[:-1])/2.
        ngtm=np.cumsum((dndm_bar*dm)[::-1])[::-1]
        ngtm = np.append(ngtm,0)
        plt.loglog(h_smt.m, ngtm, label='Analytical')

        #begin MF plot
        print ("plot MF...")
        plt.title('Mass function (z = %8.4e)'%current_redshift)
        plt.xlabel(r'$M\ [\mathrm{M_{sun}}/h]$')
        plt.ylabel(r'$N(>M)\ [h^3 \mathrm{cMpc}^{-3}]$')
        plt.legend()
        #plt.xlim(1e4,1e15)
        #plt.ylim(1e-20,1e5)
        if width_str == 'whole':
            outname = "%s_MF_whole_s%s.png"%(outname_prefix,step)
        else:
            outname = "%s_MF_%skpc_s%s.png"%(outname_prefix,width_str,step)
        print ("output {:}".format(outname))
        plt.savefig(outname)
    


######################################################
# 与えられた領域に対して複数のプロットを作成する関数
######################################################
def generate_plots(title=None, center=None,data_source=None,width_str=None):

    if data_source is None:
        data_source = ds.all_data() #** WARING ** not tested (KS TODO)
        #print ('current version of generate_plots is not compatible if data_source is None\nreturn')
        #return
        
    #nH 
    if 'nH' in pvars: 
        generate_plot("nH", plot_axis, title=title, center=center,data_source=region,width_str=width_str,halos=False,star_marker=None,file_info=infofile_fp,vmin=vmin,vmax=vmax)

    #T
    if 'T' in pvars:         
        if vmin is not None or  vmax is not None:
            generate_plot("T", plot_axis,title=title, center=center,data_source=region,width_str=width_str,velocity=False,halos=False,star_marker=None,file_info=infofile_fp,vmin=vmin,vmax=vmax)
        else:
            generate_plot("T", plot_axis,title=title, center=center,data_source=region,width_str=width_str,velocity=False,halos=False,star_marker=None,file_info=infofile_fp,vmin=1e2,vmax=1e5)
    #nHproj
    if 'nHproj' in pvars: 
        generate_plot("nHproj", plot_axis,title=title, center=center,data_source=region,width_str=width_str,halos=False,star_marker=None,file_info=infofile_fp,vmin=vmin,vmax=vmax)

    #Tproj
    if 'Tproj' in pvars:         
        generate_plot("Tproj", plot_axis,title=title, center=center,data_source=region,width_str=width_str,velocity=False,halos=False,star_marker=None,file_info=infofile_fp,vmin=vmin,vmax=vmax)

    #Sigma 
    if 'Sigma' in pvars: 
        #generate_plot("Sigma", plot_axis,title=title, center=center,data_source=region,width_str=width_str,halos=False,star_marker=None,file_info=infofile_fp,vmin=vmin,vmax=vmax)
        print ("subtraction of bulk velocity is disabled because of the error generated in class QuiverCallback(PlotCallback). I want to show density-averaged velocity but I don't know how to do so with yt.")
        #generate_plot("Sigma", plot_axis,title=title, center=center,data_source=region,width_str=width_str,halos=False,star_marker=None,file_info=infofile_fp,vmin=vmin,vmax=vmax,subtract_bulk_vel=False)
        generate_plot("Sigma", plot_axis,title=title, center=center,data_source=region,width_str=width_str,velocity=False,halos=False,star_marker=None,file_info=infofile_fp,vmin=vmin,vmax=vmax,subtract_bulk_vel=False)

    #Mach 
    if 'Mach' in pvars: 
        generate_plot("Mach", plot_axis,title=title, center=center,data_source=region,width_str=width_str,halos=False,star_marker=None,file_info=infofile_fp,vmin=vmin,vmax=vmax)

    #metal
    if 'metal' in pvars: 
        if vmin is not None or  vmax is not None:
            generate_plot("metal", plot_axis,title=title, center=center,data_source=region,width_str=width_str,velocity=False,halos=False,star_marker=None,file_info=infofile_fp,vmin=vmin,vmax=vmax)
        else:
            generate_plot("metal", plot_axis,title=title, center=center,data_source=region,width_str=width_str,velocity=False,halos=False,star_marker=None,file_info=infofile_fp,vmin=1e-6,vmax=1e0)
    #yHI
    if 'yHI' in pvars: 
        generate_plot("yHI", plot_axis,title=title, center=center,data_source=region,width_str=width_str,velocity=False,halos=False,star_marker=None,file_info=infofile_fp,vmin=vmin,vmax=vmax)

    #yHII
    if 'yHII' in pvars: 
        generate_plot("yHII", plot_axis,title=title, center=center,data_source=region,width_str=width_str,velocity=False,halos=False,star_marker=None,file_info=infofile_fp,vmin=vmin,vmax=vmax)

    #yH2
    if 'yH2' in pvars: 
        generate_plot("yH2", plot_axis,title=title, center=center,data_source=region,width_str=width_str,velocity=False,halos=False,star_marker=None,file_info=infofile_fp,vmin=vmin,vmax=vmax)

    #phi
    if 'phi' in pvars: 
        print ("** WARNING ** gravitational potential of RAMSES output is not synchronized among levels in the case with subcycles, so discontinuity may be seen on level boundaries (use non-subcycle run to plot synchronized potential)")
        generate_plot("phi", plot_axis,title=title, center=center,data_source=region,width_str=width_str,velocity=False,halos=False,star_marker=None,file_info=infofile_fp,vmin=vmin,vmax=vmax)

    #phiproj
    if 'phiproj' in pvars: 
        print ("** WARNING ** gravitational potential of RAMSES output is not synchronized among levels in the case with subcycles, so discontinuity may be seen on level boundaries (use non-subcycle run to plot synchronized potential)")
        generate_plot("phiproj", plot_axis,title=title, center=center,data_source=region,width_str=width_str,velocity=False,halos=False,star_marker=None,file_info=infofile_fp,vmin=vmin,vmax=vmax)

    #sStar
    if 'sStar' in pvars: 
        # generate_plot("sStar", plot_axis,title=title, center=center,data_source=region,width_str=width_str,halos=False,star_marker=None,file_info=infofile_fp,vmin=vmin,vmax=vmax)
        print ("subtraction of bulk velocity is disabled because of the error generated in class QuiverCallback(PlotCallback). I want to show density-averaged velocity but I don't know how to do so with yt.")
        generate_plot("sStar", plot_axis,title=title, center=center,data_source=region,width_str=width_str,velocity=False,halos=False,star_marker=None,file_info=infofile_fp,vmin=vmin,vmax=vmax,subtract_bulk_vel=False)
        #generate_plot("sStar", plot_axis,title=title, center=center,data_source=region,width_str=width_str,halos=False,star_marker=None,file_info=infofile_fp,vmin=vmin,vmax=vmax,subtract_bulk_vel=False)
        
    #sDM
    if 'sDM' in pvars: 
        # generate_plot("sDM", plot_axis,title=title, center=center,data_source=region,width_str=width_str,halos=False,DM_marker=None,file_info=infofile_fp,vmin=vmin,vmax=vmax)
        print ("subtraction of bulk velocity is disabled because of the error generated in class QuiverCallback(PlotCallback). I want to show density-averaged velocity but I don't know how to do so with yt.")
        generate_plot("sDM", plot_axis,title=title, center=center,data_source=region,width_str=width_str,velocity=False,halos=False,star_marker=None,file_info=infofile_fp,vmin=vmin,vmax=vmax,subtract_bulk_vel=False)
        #generate_plot("sDM", plot_axis,title=title, center=center,data_source=region,width_str=width_str,halos=False,star_marker=None,file_info=infofile_fp,vmin=vmin,vmax=vmax,subtract_bulk_vel=False)

    #J21
    if 'J21' in pvars: 
        if vmin is not None or  vmax is not None:
            generate_plot("J21", plot_axis,title=title, center=center,data_source=region,width_str=width_str,velocity=False,halos=False,star_marker=None,file_info=infofile_fp,vmin=vmin,vmax=vmax)
        else:
            generate_plot("J21", plot_axis,title=title, center=center,data_source=region,width_str=width_str,velocity=False,halos=False,star_marker=None,file_info=infofile_fp,vmin=1e0,vmax=1e5)
        
    #nT-diagram
    if 'nT' in pvars: 
        #generate_phase_plot('nT',data_source=region,width_str=width_str)
        generate_phase_plot('nT',data_source=region,width_str=width_str,file_info=infofile_fp,xmin=1e-6,xmax=1e8,ymin=1e0,ymax=1e9)

    #MachT-diagram
    if 'MachT' in pvars: 
        generate_phase_plot('MachT',data_source=region,width_str=width_str,file_info=infofile_fp,xmin=1e-3,xmax=1e3,ymin=1e0,ymax=1e9)

    #nmetal-diagram
    if 'nmetal' in pvars: 
        generate_phase_plot('nmetal',data_source=region,width_str=width_str,file_info=infofile_fp,xmin=1e-6,xmax=1e8,ymin=1e-8,ymax=1e1)

    #nyH2-diagram
    if 'nyH2' in pvars: 
        generate_phase_plot('nyH2',data_source=region,width_str=width_str,file_info=infofile_fp,xmin=1e-6,xmax=1e8,ymin=1e-9,ymax=1e0)

    #nyHII-diagram
    if 'nyHII' in pvars: 
        generate_phase_plot('nyHII',data_source=region,width_str=width_str,file_info=infofile_fp,xmin=1e-6,xmax=1e8,ymin=1e-7,ymax=2e0)

    #metalyH2-diagram
    if 'metalyH2' in pvars: 
        generate_phase_plot('metalyH2',data_source=region,width_str=width_str,file_info=infofile_fp,xmin=1e-8,xmax=1e1,ymin=1e-9,ymax=1e0)

    #pDM
    if 'pDM' in pvars: 
        generate_plot("pDM", plot_axis,title=title, center=center,data_source=region,width_str=width_str,halos=False,star_marker=None,velocity=False,file_info=infofile_fp,vmin=vmin,vmax=vmax)

    #DMhalo
    if 'DMhalo' in pvars: 
        generate_plot("DMhalo", plot_axis,title=title, center=center,data_source=region,width_str=width_str,velocity=False,file_info=infofile_fp,vmin=vmin,vmax=vmax)

    #pStar
    if 'pStar' in pvars: 
        #marker for star annotation
        if args.star_marker is not None: #determine from argument
            star_marker = args.star_marker
        else:                            #determine automatically
            star_marker = 'dot'     #use dot as deafult

        generate_plot("pStar", plot_axis,title=title, center=center,data_source=region,width_str=width_str,velocity=False,halos=False,star_marker=star_marker,file_info=infofile_fp,vmin=vmin,vmax=vmax)

    #MF
    if 'MF' in pvars:
        if (rank == 0):    #mass functionの作成はserial processで実行
            if not hc_found:
                print ("Halo catalog not found\nskip MF plot")
            else:
                generate_MF_plot(center=center,width_str=width_str)


######################################################
# 角運動量をを取得する関数
######################################################
def get_regional_angular_momentum_vector(region,center,bulk_vel):
    x = region['gas','x']-center[0]      #relative to center
    y = region['gas','y']-center[1]      #relative to center
    z = region['gas','z']-center[2]      #relative to center
    mass = region['gas','cell_mass']
    vx = region['gas','velocity_x']-bulk_vel[0]  #subtract bulk velocity
    vy = region['gas','velocity_y']-bulk_vel[1]  #subtract bulk velocity
    vz = region['gas','velocity_z']-bulk_vel[2]  #subtract bulk velocity

    Jx = ((y[:]*vz[:] - z[:]*vy[:])*mass[:]).sum()
    Jy = ((z[:]*vx[:] - x[:]*vz[:])*mass[:]).sum()
    Jz = ((x[:]*vy[:] - y[:]*vx[:])*mass[:]).sum()   

    Jam = np.array([Jx,Jy,Jz]) #angular momentum vector

    return Jam


#--------------- オブジェクトの取得 -----------------#

#------- wholeの場合 -------#
if object_type == "sim":
    #全領域で計算
    print ('\n** plot around simulation box center')
    #計算領域の中心 in code_unit
    pos_obj = yt.YTArray([0.5,0.5,0.5],input_units='code_length',registry = ds.unit_registry)
    

#------- halo/dmax/diskの場合 -------#
#elif object_type == "halo":
elif object_type in ["halo","dmax","disk"]:    
    #(i_obj)番目に重たいハローを取得
    ad_hc=hc.halos_ds.all_data()
    Nhalo = len(ad_hc['particle_mass'])
    if i_obj > Nhalo:
        print ("i_obj = {:} but the total nubmer of halo = {:}\nexiting...".format(i_obj,Nhalo))
        sys.exit()    

    arg_halo=np.argsort(ad_hc['particle_mass'])[-(i_obj)] # 対応するハローのindex
    mass_obj=ad_hc['particle_mass'][arg_halo].in_units('Msun')
    pos_obj=ad_hc['particle_position'][arg_halo]
    rvir_obj=ad_hc['virial_radius'][arg_halo].in_units('kpc')

    #halos_dsのcode_lengthが1となっていて、単純にcenter=pos_objとすると不整合が生じるのでunit registryを変更
    #他の量については、plot時の影響は無さそうなのでこのままでok
    pos_obj = yt.YTArray(pos_obj.in_units('cm'),input_units='cm',registry = ds.unit_registry).in_units('code_length')

    print ('\n** plot around %d th most massive halo **'%(i_obj))
    print ('mass: %4.2e Msun'%(mass_obj))    
    print ('position: ', pos_obj)
    print ('virial radius: ', rvir_obj)

    #------- dmaxの場合 -------#
    #プロットの中心をhalo virial半径内の質量最大の点に変更
    if object_type == "dmax":    
        radius = rvir_obj
        region = ds.sphere(pos_obj,radius)
        nHmax = region.max('H_nuclei_density')
        #中心を密度最大点に設定
        pos_obj = yt.YTArray(region.argmax('H_nuclei_density')).in_units('code_length')

        print ('plot center is set to the position of density maximum: ', pos_obj, ', maximum density: ', nHmax)

        #プロットの中心をhalo finderでの値からhalo virial半径内の重心に変更
    elif args.halo_center == 'bary':     
        radius = rvir_obj
        region = ds.sphere(pos_obj,radius)

        mass = region['DM','particle_mass']
        posx = region['DM','particle_position_x']
        posy = region['DM','particle_position_y']
        posz = region['DM','particle_position_z']        

        m_halo = mass[:].sum()
        x_gc = (posx[:]*mass[:]).sum()/m_halo
        y_gc = (posy[:]*mass[:]).sum()/m_halo
        z_gc = (posz[:]*mass[:]).sum()/m_halo

        #中心を重心に設定
        pos_obj = yt.YTArray([x_gc,y_gc,z_gc])

        print ('plot center is set to the position of gravity center: ', pos_obj, ', halo mass: ', m_halo.in_units('Msun'))


#------- starの場合 -------#
elif object_type == "star":
    # idp = i_obj の星を取得 (idp is particle identification number used in Ramses, PopIIおよびPopIIIは生まれた順にidpで番号付けされている)
    idp_list =  np.array(ad['star','particle_identity'].tolist() + ad['POPIII','particle_identity'].tolist() + 
                         ad['supernova','particle_identity'].tolist() + ad['dead','particle_identity'].tolist()
                         + ad['BH','particle_identity'].tolist(),dtype=int).tolist()
    
    if i_obj not in idp_list:
        print ("i_obj = {:} is not in the idp list = {:}\nexiting...".format(i_obj,idp_list))
        sys.exit()    

    for star_type_cand in ['star', 'POPIII', 'supernova', 'dead', 'BH']:
        if i_obj in np.array(ad[star_type_cand,'particle_identity'].tolist(),dtype=int):
            star_type = star_type_cand
            arg_star = np.array(ad[star_type_cand,'particle_identity'].tolist(),dtype=int).tolist().index(i_obj)
            break
    
    #星の情報を取得
    pos_obj = ad[star_type,'particle_position'][arg_star]
    be_obj = ad[star_type,'particle_birth_epoch'][arg_star]
    birth_redshift_obj = codetime_to_redshift(float(be_obj))
    mass_obj = ad[star_type,'particle_mass'][arg_star].in_units('Msun')

    print ('\n** plot around a %s with idp = %d **'%(star_type,i_obj))
    print ('mass: %4.2e M_sun'%mass_obj)
    print ('position: ', pos_obj)
    print ('birth epoch: t = ', be_obj, ', z = ', birth_redshift_obj)

#-------------------------- ここからプロット作成 ------------------------------#
#general set-up
#plt.rcParams["font.family"] = "Times New Roman" #よくわからないが、たまに勝手に太字になってしまうのでひとまずデフォルトのフォントでいく

center = pos_obj
#プロットのタイトルに書き込む情報
if object_type == "sim":
    title =  r"$[{:6.4f}, {:6.4f}, {:6.4f}], \mathrm{{\#}}{:}$".format(
        float(pos_obj[0]),float(pos_obj[1]),float(pos_obj[2]),step)
if object_type == "halo":
    title =  r"$\mathrm{{halo}}\ {:}: {:.2g}\ M_\odot, {:3.1f}\ \mathrm{{kpc}}, [{:6.4f}, {:6.4f}, {:6.4f}], \mathrm{{\#}}{:}$".format(
        i_obj,float(mass_obj),float(rvir_obj),float(pos_obj[0]),float(pos_obj[1]),float(pos_obj[2]),step)
if object_type == "dmax":
    title =  r"$\mathrm{{densmax\ of\ halo}}\ {:}: {:3.1f} \mathrm{{cm^{{-3}}}}, {:.2g}\ M_\odot, {:3.1f}\ \mathrm{{kpc}}, [{:6.4f}, {:6.4f}, {:6.4f}], \mathrm{{\#}}{:}$".format(
        i_obj,float(nHmax),float(mass_obj),float(rvir_obj),float(pos_obj[0]),float(pos_obj[1]),float(pos_obj[2]),step)
if object_type == "disk":
    title =  r"$\mathrm{{halo w/ disk polar axis}}\ {:}: {:.2g}\ M_\odot, {:3.1f}\ \mathrm{{kpc}}, [{:6.4f}, {:6.4f}, {:6.4f}], \mathrm{{\#}}{:}$".format(
        i_obj,float(mass_obj),float(rvir_obj),float(pos_obj[0]),float(pos_obj[1]),float(pos_obj[2]),step)    
elif object_type == "star":
    title =  r"$\mathrm{{idp}}\ {:}: \mathrm{{{:}}}, {:4.1f}\ M_\odot, z_\mathrm{{form}} = {:6.2f}, [{:6.4f}, {:6.4f}, {:6.4f}], \mathrm{{\#}}{:}$".format(
        i_obj,star_type,float(mass_obj),birth_redshift_obj,float(pos_obj[0]),float(pos_obj[1]),float(pos_obj[2]),step)

#width_listに含まれる各領域サイズでそれぞれプロットを作成
for width_str in args.width_list.split():
    if width_str == 'whole':
        width = boxsize_in_kpc
    else:
        if width_str[0]=='0':                          #0から始まる場合は小数
            n_zeros = width_str.count('0')            #003 -> 2
            width = ds.quan(float(width_str)*10**(-n_zeros), "kpc") #003 -> 0.03
        else:                                         #通常の場合                             
            width = ds.quan(float(width_str), "kpc")  #30 -> 30
            
    if rank == 0:       
        print ("generating plot for width ", width)

    #よくわからないが、widthをcode_unitにしないとバグが発生する場合があるかも
    width = width.in_units('code_length')

    #------------ オブジェクト周りのregionを取得 --------------------#
    # #data for box region (ギリギリだとときどきはみ出るので修正必要)
    # left_edge = np.ones(3)* (center - 0.5*width) #boxの左下を定義
    # right_edge = np.ones(3)* (center + 0.5*width) #boxの右上を定義
    # region = ds.box(left_edge,right_edge)


    #------- disk以外の場合 -------#
    if object_type != "disk":
        #data for box region (ギリギリだとときどきはみ出るのでちょっと大きめにとってみる)
        if rank == 0:       
            print ("(KS WARNING) set slighlty larger box compared with given width")
        left_edge = np.ones(3)* (center - 0.51*width) #boxの左下を定義
        right_edge = np.ones(3)* (center + 0.51*width) #boxの右上を定義
        # #もう少し大きめにとってみる？？ (少しプロットがミスリーディングになるのでひとまずやめておく)
        # left_edge = np.ones(3)* (center - 0.6*width) #boxの左下を定義
        # right_edge = np.ones(3)* (center + 0.6*width) #boxの右上を定義    
        region = ds.box(left_edge,right_edge)    

    #------- diskの場合 -------#
    else:
        #data for spherical region (containg a box with width on a side)
        radius = np.sqrt(3.)/2.*width
        region = ds.sphere(center,radius)



    #------- diskの場合 -------#
    if object_type == "disk":
        #polar_axisを領域内の角運動量ベクトルの向きとして取得
        bulk_vel = region.quantities.bulk_velocity().in_units("km/s")
        Jam = get_regional_angular_momentum_vector(region,center,bulk_vel) #angular momentum of the region
        polar_axis = Jam/np.linalg.norm(Jam)
        print ("bulk_vel, center, polar_axis:", bulk_vel, center, polar_axis)


    #----------------------------
    #get particle info
    #----------------------------
    DM = region['DM','particle_mass'].in_units('Msun')                 #DM particles
    pop2 = region['star','particle_mass'].in_units('Msun')             #Pop II stars
    pop3 = region['POPIII','particle_mass'].in_units('Msun')           #living Pop III stars
    supernova = region['supernova','particle_mass'].in_units('Msun')   #Pop III stars taking place SNe
    dead = region['dead','particle_mass'].in_units('Msun')             #Pop III stars after SNe
    BH = region['BH','particle_mass'].in_units('Msun')                 #Pop III remnant BHs
    SFC = region['SFC','particle_mass'].in_units('Msun')               #Star-Forming Clouds
    PSC = region['PSC','particle_mass'].in_units('Msun')               #Passive Stellar Clusters

    #particle number
    N_DM = len(DM)
    N_pop2 = len(pop2)
    N_pop3 = len(pop3)
    N_sn = len(supernova)
    N_dead = len(dead)
    N_BH = len(BH)
    N_SFC = len(SFC)
    N_PSC = len(PSC)    

    #mass
    M_DM = DM.sum()
    M_pop2 = pop2.sum()
    M_pop3 = pop3.sum()
    M_sn = supernova.sum()
    M_dead = dead.sum()
    M_BH = BH.sum()  #sink粒子がmassを保持しているのでBH粒子はテスト粒子（位置と速度はだいたい一致すると期待）

    if rank == 0:
        # print ('Number of particles (DM, Pop II, Pop III, SN, dead, BH): {:}, {:}, {:}, {:}, {:}, {:}'.format(
        #     N_DM, N_pop2, N_pop3,N_sn,N_dead,N_BH), flush=True)
        # print ('Mass of particles (DM, Pop II, Pop III, SN, dead, BH): {:.2e}, {:.2e}, {:.2e}, {:.2e}, {:.2e}, {:.2e}'.format(
        #        M_DM, M_pop2, M_pop3,M_sn,M_dead,M_BH), flush=True)
        print ('Number of particles (DM, Pop II, Pop III, BH, SFC, PSC): {:}, {:}, {:}, {:}, {:}, {:}'.format(
            N_DM, N_pop2, N_pop3, N_BH, N_SFC, N_PSC), flush=True)        
        print ('Mass of particles (DM, Pop II, Pop III): {:.2e}, {:.2e}, {:.2e}'.format(
               M_DM, M_pop2, M_pop3), flush=True)

    #データの解像度が低すぎてセル数が少ない場合、エラーが発生することがあったのでプロットをスキップ
    try:
        #Ncell = len(region['ramses','x'])
          Ncell = len(region['gas','x'])        #DM-only計算のとき('ramses','x')は存在しないが、('gas','x')なら存在する(?)
    except ValueError as error:
        import traceback
        traceback.print_exc()
        print('Error occured when obtaining Ncell. This is maybe due to a too small region.\nskip plot for width_str = {:}'.format(width_str))
        continue
    print ('Number of the cells: {:}'.format(Ncell))
    if Ncell < 4**3:
        print ('Number of the cells in the region is too small\nskip plot for width_str = {:}'.format(width_str))
        continue

    #複数のプロットを作成
    generate_plots(title=title, center=center,data_source=region,width_str=width_str)


# #for debug
# yt.show_colormaps(subset=['algae', 'kamae', 'Spectral','jet','hot',
#                               'arbre', 'dusk', 'octarine', 'kelp','winter', 'aaa'],
#                       filename="yt_native.png")
