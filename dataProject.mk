TARGET_NAME:=PerlinCuda
TARGET_MODE:=EXE
TARGET_PATH:=/tmp/ramdrive/${USER}/Targets/${TARGET_NAME}
TARGET_DEPLOY_PATH:=#../PRODUCTION/Deploy/bin
CODE_DEFINE_VARIABLES+=
SRC_AUX:=  ../PRODUCTION/commonExt/cuda/  ../PRODUCTION/commonExt/include/
NVCCFLAGS= -O3 -use_fast_math -arch=sm_20
NVCCLDFLAGS =
