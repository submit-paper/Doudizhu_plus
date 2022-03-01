checkpoint_dir=$1

landlord_path=$landlord_dir`ls -v "$checkpoint_dir"landlord_weights* | tail -1`
landlord_up_path=$landlord_up_dir`ls -v "$checkpoint_dir"landlord_up_weights* | tail -1`
landlord_down_path=$landlord_down_dir`ls -v "$checkpoint_dir"landlord_down_weights* | tail -1`
pre_landlord_path=$pre_landlord_dir`ls -v "$checkpoint_dir"pre_landlord_weights* | tail -1`
pre_landlord_up_path=$pre_landlord_up_dir`ls -v "$checkpoint_dir"pre_landlord_up_weights* | tail -1`
pre_landlord_down_path=$pre_landlord_down_dir`ls -v "$checkpoint_dir"pre_landlord_down_weights* | tail -1`

echo $landlord_path
echo $landlord_up_path
echo $landlord_down_path
echo $pre_landlord_path
echo $pre_landlord_up_path
echo $pre_landlord_down_path


mkdir -p most_recent_model

cp $landlord_path most_recent_model/landlord.ckpt
cp $landlord_up_path most_recent_model/landlord_up.ckpt
cp $landlord_down_path most_recent_model/landlord_down.ckpt
cp $pre_landlord_path most_recent_model/pre_landlord.ckpt
cp $pre_landlord_up_path most_recent_model/pre_landlord_up.ckpt
cp $pre_landlord_down_path most_recent_model/pre_landlord_down.ckpt
