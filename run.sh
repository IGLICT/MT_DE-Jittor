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

# run
if [ $do_test == 0 ]; then
python train.py \
	--name $trainname \
	--nThreads 2 \
	--dataroot $datasetname \
	--n_blocks_global 6 \
	--ngf 32 \
	--ndf 32 \
	--norm instance \
	--batchSize 1 \
	--resize_or_crop scale_width \
	--loadSize 512 \
	--fineSize 512 \
	--save_latest_freq 5000 \
	--display_freq 10 \
	--print_freq 10 \
	--no_flip \
	--tf_log \
	--no_ps_loss \
	--num_of_frame 2 \
	--input_type 0 \
	--usemulti \
	--gpu_ids $gpu_ids
#	--continue_train
	# --resize_or_crop scale_width \
	# --continue_train
	#--lambda_gan_feat 100 \
	# --no_ganFeat_loss 
fi