from subprocess import call

class S3Utils:

    @staticmethod
    def upload(model_type):
        call("cp /data/config /root/.aws/.".split(sep=' '))
        call("cp /data/credentials /root/.aws/.".split(sep=' '))

        aws_cmd = "aws s3 cp --recursive /ekholabs/toxicity/model_output/%s s3://ekholabs-kaggle-models" % model_type
        call(aws_cmd.split(sep=' '))