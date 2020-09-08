
# NoisyArt Dataset
NoisyArt is dataset designed to support research on webly-supervised recognition of artworks. It was also been designed to support multi-modality learning and zero-shot learning, thanks to it's multi-modal nature.
The dataset consists of more than 80'000 webly-supervised images from 3120 classes, and a subset of 200 classes with more than 1300 verified images. Text and metadata for each class is also provided, to support zero-shot learning and  other multi-modality techniques in general.


# Data
We used DBpedia as main sources of metadata, retrieving textual informations and images for 3120 artworks scattered around the world. We created queried Google Images and Flickr to retrieve more images for each artwork.
A test set with verified images was created using a subset of 200 classes.
In the next table is is shown a panoramic on the number of classes and images for different splits of the dataset.

<p align="center">
  <img src="https://github.com/delchiaro/NoisyArt/raw/readme/readme-assets/imgs/noisyart-class-split.png" width="600" title="Some images taken from the dataset">
</p>


## Images
Breafly, for each class we have the following images:
* 20: images retrieved from Google Images (could be less because of some corrupted images we retrieved).
* [0, 12]: images retrieved from Flickr.
* [0, 1]: seed image retrieved from DBpedia/Wikipedia.

<p align="center">
  <img src="https://github.com/delchiaro/NoisyArt/raw/readme/readme-assets/imgs/dataset-images.png" width="700" title="Some images taken from the dataset">
</p>


## Metadata
For each artwork we got metadata related to the artist, artwork itself and the museum/location where is preserved. We stored informations like:
* Artwork title, comment, description and creation location.
* Authors data with the following information for each one: name, comment, artistic movement, birth date, death date, birth location, nationality.
* Museums data with the name and the location.

Moreover, for each of these 3 entities we store a DBpedia URI that can be used to retrieve more data using others SPARQL queries or, manually, using a web browser.

<p align="center">
  <img src="https://github.com/delchiaro/NoisyArt/raw/readme/readme-assets/imgs/json-example.png" width="700" title="JSON metadata for one of the 3120 artworks in the dataset.">
</p>

## Noise and Bias
There are mainly 3 kind of noise in this dataset, due to its webly nature:

* **Outliers**: some images retrieved from the web could be representation of concept complitely differents from the searched one, and complitely alien from the dataset topic (picture of artworks). This kind of noise is tipical of images retrived from the web. We notice a high quantity of this noise in images taken from Flickr, and some in images from Google.

* **Labelflip**: Some artists made different versions of the same arwork with the same title, but are actually two different classes (e.g.: Rembrandt self-portraits). Also, some legendary scenes were depicted more and more times from different artists in the history (e.g.: Saint George and the Dragon, Madonna and child, etc..). For these reasons the search results for some artworks could present a variable quantity of images that it's not what we wanted, and we ends up with some classes having the images of the others (wrong labelling of the images or "labelflip" noise).

* **Image-domain Bias (low diversity)**: "Google is biased" is a famous sentence that sums up the a phenomenon common to all search engines: if we search for a specific concept on the web, we ends up retriving iconic representation of that concept. In the case of artworks this problem is more observable for paintings: searching on google for a painting will ends up with a lot of pictures that are more similar to scans than photos. This is probably what a normal Google Image user would like to see, but it's a problem for our target application because we collect a lot of similar pictures with low diversity, that doesn't bring new informations that can be learned from the classifier.

* **Label-domain Bias (labelflip)**: The same bias of the search engines, that try to show first the most iconic representation of a concept, can also bring some more labelflip noise in our instance-recognition dataset. Infact, if we try to search on Google Images for a not-so-famous artwork made by a famous artist, we will receive a lot of images of iconic artworks from the same artist. Try yourself to search "Anxiety Munch" on Google images: in the results you'll also see a lot of images of "The Scream", that is one of the most iconic artwork made by the same artist. If you search for "The Scream Munch" you'll hardly see an image of Anxiety artwork.

# Processed Data

## Pretrained CNNs image features
We used 5 differents CNNs pretrained on ImageNet to extract visual features from all the images in the dataset:
* VGG16, VGG19
* ResNet50, ResNet101, ResNet152

Those feature vectors will be released publicly.



## Pretrained doc2vec textual features
We trained a doc2vec over the whole wikipedia dump and we used the trained model to get a dense feature vector for each class, processing the description of the artwork togheter with some other informations taken from the metadata (artist informations like name, description and artistic movement). 

All the informations will be released publicly:
* All the retrieved metadata per each class (usefull to create again new textual document associated with each class).
* Our textual documents created ad-hoc per each class.
* The processed doc2vec features obtained processing each class document.


# Experiments and Results

We used the pre-processed image features to train a baseline classifier (BL) with some additional techniques to cope with the described noise. The main techniques we presents are the following:

* **Labelflip Absorption Layer (LF)** [Sukhbaatar et al., 2014]: use a new fully connected layer without bias after the final softmax output. The weight matix is square and initialized to diagonal matrix. After some epochs the weights are unlocked and trained allowing this layer to model class confusion probabilities thanks also to a trace regularization. This layer should absorb part of the labelflip noise leaving the network free to learn on "clean" labels.

* **Entropy Scaling (ES)**: class-normalized entropy of a training sample is used as an indicator of how confident the model is about a particular input sample. We use this indicator to weight samples in input to the network, trying to lowering the loss for images for which the model is less confident.

* **Gradual Bootstrapping (BS)**: we apply Entropy Scaling using images from DBpedia or the first image from Google as an high-confident example for the class. We perform a pre-training for few epochs on these images and after that we compute the entropy-scaling score for all the other images in the training-set, so that the more similar to the seeds should be more important in the beginning of the training while keeping the loss contribution from the hard-images and outlier low. Entropy scaling is re-computed every few epochs: the hope is that hard-images loss contribution will rise at each entropy-scaling computation while keeping low the loss contribution from outliers.


<p align="center">
  <img src="https://github.com/delchiaro/NoisyArt/raw/readme/readme-assets/imgs/results.png" width="400" title="Results from the paper">
</p>


# Downloads
The dataset structure with metadata and without images can be downloaded from the release tab in this repository.

The following table contains links to dataset images.
You can download images in their original format and resolution, or image resized keeping the original aspect ratio with the shorter dimension resized to 255, stored with a lossless format (PNG).
Each row is a different split of the dataset.

<table>
  		<tr> 
  			<th>Split</th>
  			<th>Original</th>
  			<th>Resize-255 (png)</th>
  		</tr>
  		<tr> 
  			<td>Trainval 3120 classes</td>
  			<td><a href='https://www.micc.unifi.it/wp-content/uploads/datasets/noisyart/noisyart_trainval_3120.zip'>link</a></td>
  			<td><a href='https://www.micc.unifi.it/wp-content/uploads/datasets/noisyart/noisyart_trainval_3120_r255_png.zip'>link</a></td>
  </tr>
		<tr> 
  			<td>Trainval 200 classes</td>
  			<td><a href='https://www.micc.unifi.it/wp-content/uploads/datasets/noisyart/noisyart_trainval_200.zip'>link</a></td>
  			<td><a href='https://www.micc.unifi.it/wp-content/uploads/datasets/noisyart/noisyart_trainval_200_r255_png.zip'>link</a></td>
  		</tr>
  		<tr> 
  			<td>Test-set 200 classes</td>
  			<td><a href='https://www.micc.unifi.it/wp-content/uploads/datasets/noisyart/noisyart_test_200.zip'>link</a></td>
  			<td><a href='https://www.micc.unifi.it/wp-content/uploads/datasets/noisyart/noisyart_test_200_r255_png.zip'>link</a></td>
  		</tr>
  </table
	
MD5SUM:
<table>
    <tr> 
        <td>noisyart_test_200_r255_png.zip</td>
	<td>146ea3d21ffe04d1b7182802cfdab23a</td>    
    </tr>
    <tr> 
        <td>noisyart_test_200.zip</td>
	<td>02fc6c55e72ef1e57a0d9aaa1ddbc33c</td>    
    </tr>
    <tr> 
        <td>noisyart_trainval_200_r255_png.zip</td>
	<td>926da12d1387202320a1fd48da9aca6c</td>    
    </tr>
    <tr> 
        <td>noisyart_trainval_200.zip</td>
	<td>ae020565f3047f57580cca31dabfe5aa</td>    
    </tr>
    <tr> 
        <td>noisyart_trainval_3120_r255_png.zip</td>
	<td>f42bbf7f853cc81048db96e3cb57e74c</td>    
    </tr>
    <tr> 
        <td>noisyart_trainval_3120.zip</td>
	<td>142b6e33a0f5398b23c7c7d372edce58</td>    
    </tr>
</table>

### New split for CMU Oxford Sculpture Dataset 
A new split is available to conduct artwork instance recognition experiments on CMU Oxford Sculpture Dataset.
This split is published in the releases of this repo: [new split](https://github.com/delchiaro/NoisyArt/releases/tag/cmu-split-1.0)



# bibtex
```
@inproceedings{DelChiaro2019,
author = {Chiaro, R Del and Bagdanov, A and Bimbo, A Del},
booktitle = {Proceedings of the 14th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications},
doi = {10.5220/0007392704670475},
file = {:home/delchiaro/papers/Chiaro, Bagdanov, Bimbo - 2019 - {\{}NoisyArt{\}} A Dataset for Webly-supervised Artwork Recognition.pdf:pdf},
publisher = {{\{}SCITEPRESS{\}} - Science and Technology Publications},
title = {{{\{}NoisyArt{\}}: A Dataset for Webly-supervised Artwork Recognition}},
url = {https://doi.org/10.5220/0007392704670475},
year = {2019}
}
```



# Acknowledgments
The authors of this work would like to thank Nvidia Corporation for the donation of the Titan XP GPU used in this research.
