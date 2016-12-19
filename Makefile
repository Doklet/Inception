current_dir=$(shell pwd)
data_dir=${current_dir}/data
apple_dir=/Users/marcusnilsson/Pictures/apple

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib:/usr/local/cuda/extras/CUPTI/lib
export CUDA_HOME=/usr/local/cuda
export PATH=/Developer/NVIDIA/CUDA-8.0/bin:$PATH 
export DYLD_LIBRARY_PATH=/Developer/NVIDIA/CUDA-8.0/lib:$DYLD_LIBRARY_PATH

# Mac OS X, CPU only, Python 2.7:
# export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.11.0-py2-none-any.whl
# Mac OS X, GPU enabled, Python 2.7:
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/gpu/tensorflow-0.11.0-py2-none-any.whl

install_tf:
	pip install --upgrade ${TF_BINARY_URL}

install_requirements:
	pip install -r requirements.txt

clean:
	rm -rf data

test_data_flower:
	-mkdir data
	wget -c -N http://download.tensorflow.org/example_images/flower_photos.tgz
	-mkdir ${data_dir}/flower_photos
	tar xfp flower_photos.tgz --directory=${data_dir};
	chmod -R 777 ${data_dir}
	rm -rf ${data_dir}/flower_photos/daisy
	rm -rf ${data_dir}/flower_photos/dandelion
	rm -rf ${data_dir}/flower_photos/sunflowers

test_data_apple:
	-mkdir data
	cp -Rv ${apple_dir} ${data_dir}/apple

setup_env:
	source ../ENV/bin/activate

retrain:
	$(shell python retrain.py \
	--bottleneck_dir=${data_dir}/bottlenecks \
	--how_many_training_steps 500 \
	--model_dir=${data_dir}/inception \
	--output_graph=${data_dir}/retrained_graph.pb \
	--output_labels=${data_dir}/retrained_labels.txt \
	--image_dir=${data_dir}/apple)

inference:
	python label_image.py ${data_dir}/apple/unseen/good_thumb_IMG_0618_1024.jpg

start:
	python server.py

stage:
	-rm ./static
	echo 'Linking ui $(CURDIR)/../classify-ui/app'
	ln -s $(CURDIR)/../classify-ui/app ./static

