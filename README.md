# Kaggle - Toxicity Challenge

This repositoru aims to share some of the code I wrote for the [Toxicity Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge).
I did not make it to the first positions, but getting access to the dataset and being able to try it out on AWS GPU
instances was worth it.

The datasets, for both training and testing, are not available in this repository, but they can be easily download at
the Toxicity data page: [datasets](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)

The model built here uses an activation function I have developed myself, which was part of the IBM Watson AI XPRIZE
competition. The function has demonstrated to be better than the ReLU function. More about the ReLUs function (as I call it),
will soon be available in a separate paper, as there is a hyper parameter that needs to be tuned depending on the network architecture.

## Performance

### MacBook Pro

On a MacBook Pro, with 16GB, 4 cores, Intel i7, one epoch takes about 30 minutes.

### AWS

On a g2.2xlarge GPU Instance, with 15GB and 8 vCPUs, one epoch takes about 3 minutes. Quite impressive!

# Running the Model

## Running Locally

You are not encouraged to run it locally as you would have to go through a dependency hell. Instead, just go to the next section and try it out with Docker!

## Running with Docker

You are expected to have a Docker engine installed on your MacBook or laptop.

1. ```docker run -d -v [path_to_hyperparams.json]:/data ekholabs/toxicity```
  * There is an example file under the ```examples``` directory.

There might be some errors after the execution is done, as the code is expected to connect to a AWS S3 bucket to transfer 
the best model weights and results.

More information on running it with Terraform can be found in the [Automated ML](https://github.com/ekholabs/automated_ml) repository I created.