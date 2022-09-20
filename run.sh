srcaim=${1}
trgaim=${2}
gpu_ids=0
do_test=1

datadir="/home/sunyangtian/dance_datasets"
codedir="/mnt/b/sunyangtian/MTDEnet_jittor"

#create dataset link
# trainname="${srcaim}2${trgaim}_blend_100_First"
# datasetname="datasets_${srcaim}2${trgaim}_for205"

trainname="${srcaim}2${trgaim}_blend_100_First"
datasetname="datasets_${srcaim}2${trgaim}_for205"
if [ ! -d $datasetname ]; then
  echo "make dataset"
  mkdir $datasetname
  #train
  ln -s "${datadir}/${trgaim}/all_512_norm"            "${codedir}/${datasetname}/train_trg_img"
  ln -s "${datadir}/${trgaim}/img_obj_smooth_512_norm" "${codedir}/${datasetname}/train_trg_densepose"
  ln -s "${datadir}/${trgaim}/openpose_img_512_norm"   "${codedir}/${datasetname}/train_trg_openpose"
  ln -s "${datadir}/${srcaim}/all_512_norm"            "${codedir}/${datasetname}/train_src_img"
  ln -s "${datadir}/${srcaim}/img_obj_smooth_512_norm" "${codedir}/${datasetname}/train_src_densepose"
  ln -s "${datadir}/${srcaim}/openpose_img_512_norm"   "${codedir}/${datasetname}/train_src_openpose"
  # ln -s "${datadir}/${srcaim}/bg.jpg"   "${codedir}/${datasetname}/src_bg.jpg"
  # ln -s "${datadir}/${trgaim}/bg.jpg"   "${codedir}/${datasetname}/trg_bg.jpg"
  #test
  ln -s "${datadir}/${trgaim}/all_512_norm"            "${codedir}/${datasetname}/test_trg_img"
  ln -s "${datadir}/${trgaim}/img_obj_smooth_512_norm" "${codedir}/${datasetname}/test_trg_densepose"
  ln -s "${datadir}/${trgaim}/openpose_img_512_norm"   "${codedir}/${datasetname}/test_trg_openpose"
  ln -s "${datadir}/${srcaim}/all_512_norm"            "${codedir}/${datasetname}/test_src_img"
  ln -s "${datadir}/${srcaim}/img_obj_smooth_512_norm" "${codedir}/${datasetname}/test_src_densepose"
  ln -s "${datadir}/${srcaim}/openpose_img_512_norm"   "${codedir}/${datasetname}/test_src_openpose"
fi
