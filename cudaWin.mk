# Version 	: 0.0.1
# Date		: 26.01.2012
# Author 	: Cedric.Bilat@he-arc.ch
#
# Contrainte
#
#	Dans les definitions de variables ci-dessous, ne laisser jamais d'espace a la fin!!!
#

##############
#   public   #
##############


#########
# Visual #
#########

#separateur est espace,sans extension .a, sans prefixe lib
EXCLUDE_LIBRARY_FILES:=
ADD_LIBRARY_FILES:=

#Compilation: (minimal vide)
#/DWIN32 : only for compiling cppTest
CXXFLAGS  := " /DWIN32 /openmp"

#Linkage: (minimal vide)
CXXLDFLAGS := 

#" " necessaire!
#flag seront surcharger par la suite

##############
#   pivate   #
##############

COMPILATEUR:=nvidia
OS:=Win
USER=${USERNAME}

#chargement de la makeFile principale
-include dataProject.mk
-include ../PRODUCTION/MANAGER/makefile/private/makeFileCudaWin.mk
 
##############
#   END      #
##############
