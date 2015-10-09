#!/bin/sh

./dfield_exec.sh -5 22 ../../../Data/Uncertainty/heart/heart.pdata ../../../Data/Uncertainty/heart/heart.point ../../../Data/Uncertainty/heart/heart.btet heart.df
./dfield_exec.sh -4 20 ../../../Data/Uncertainty/heart/heart_isotropic.pdata ../../../Data/Uncertainty/heart/heart.point ../../../Data/Uncertainty/heart/heart.btet heart_isotropic.df
./dfield_exec.sh -9 17 ../../../Data/Uncertainty/heart/heart_v2.pdata ../../../Data/Uncertainty/heart/heart.point ../../../Data/Uncertainty/heart/heart.btet heart_v2.df
./dfield_exec.sh -9 17 ../../../Data/Uncertainty/heart/heart_v3.pdata ../../../Data/Uncertainty/heart/heart.point ../../../Data/Uncertainty/heart/heart.btet heart_v3.df

./dfield_exec.sh -4 16 ../../../Data/Uncertainty/pcc/pcc1.pdata ../../../Data/Uncertainty/pcc/pcc.point ../../../Data/Uncertainty/pcc/pcc.bhex pcc1.df
./dfield_exec.sh -4 15 ../../../Data/Uncertainty/pcc/pcc1_0.pdata ../../../Data/Uncertainty/pcc/pcc.point ../../../Data/Uncertainty/pcc/pcc.bhex pcc1_0.df
./dfield_exec.sh -4 15 ../../../Data/Uncertainty/pcc/pcc1_90.pdata ../../../Data/Uncertainty/pcc/pcc.point ../../../Data/Uncertainty/pcc/pcc.bhex pcc1_90.df

./dfield_exec.sh -4 14 ../../../Data/Uncertainty/pcc/pcc2.pdata ../../../Data/Uncertainty/pcc/pcc.point ../../../Data/Uncertainty/pcc/pcc.bhex pcc2.df
./dfield_exec.sh -4 14 ../../../Data/Uncertainty/pcc/pcc2_0.pdata ../../../Data/Uncertainty/pcc/pcc.point ../../../Data/Uncertainty/pcc/pcc.bhex pcc2_0.df
./dfield_exec.sh -3 14 ../../../Data/Uncertainty/pcc/pcc2_90.pdata ../../../Data/Uncertainty/pcc/pcc.point ../../../Data/Uncertainty/pcc/pcc.bhex pcc2_90.df

./dfield_exec.sh -3 14 ../../../Data/Uncertainty/slab/slab0.pdata ../../../Data/Uncertainty/slab/slab.point ../../../Data/Uncertainty/slab/slab.bhex slab0.df
./dfield_exec.sh -4 14 ../../../Data/Uncertainty/slab/slab120.pdata ../../../Data/Uncertainty/slab/slab.point ../../../Data/Uncertainty/slab/slab.bhex slab120.df

