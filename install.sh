curl https://s3.amazonaws.com/my89-frame-annotation/public/of500_images_resized.tar > resized.tar
tar -xvf resized.tar
mv of500_images_resized resized_256

mkdir baseline_models
curl https://s3.amazonaws.com/my89-frame-annotation/public/baseline_encoder > baseline_models/baseline_encoder
curl https://s3.amazonaws.com/my89-frame-annotation/public/baseline_resnet_101 > baseline_models/baseline_resnet_101
curl https://s3.amazonaws.com/my89-frame-annotation/public/baseline_resnet_50 > baseline_models/baseline_resnet_50
curl https://s3.amazonaws.com/my89-frame-annotation/public/baseline_resnet_34 > baseline_models/baseline_resnet_34
