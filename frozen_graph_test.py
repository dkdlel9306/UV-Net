import os
import glob
import cv2
import time
import helpers
import utils
import numpy as np
import tensorflow as tf
import win_unicode_console

from tensorflow.python.client import timeline

win_unicode_console.enable()

CWD_PATH = os.getcwd()
DATASET_NAME = 'UV_dataset'
LABEL_NAME = 'class_dict.csv'

PATH_TO_TEST_IMAGES_DIR = 'D:/conference/ICCV2019/ICCV 2019/data/Test/cropface'
TEST_IMAGE_PATH = os.path.join(CWD_PATH, PATH_TO_TEST_IMAGES_DIR)
RESULT_PATH = 'D:/conference/ICCV2019/ICCV 2019/networks/unet_yolov3/result'

"""
frozen_uv_model : Placeholder ~ logits/2D
frozen_mode : Placeholder ~ Mean
"""
GRAPH_NAME = 'D:/conference/ICCV2019/ICCV 2019/networks/unet_yolov3/frozen_unet_yolov3.pb'
#GRAPH_PATH = os.path.join(CWD_PATH, GRAPH_NAME)

START_COUNT = 1
WIDTH = 512
HEIGHT = 512
#RESIZE_HEIGHT = int(3264 / (2448 / WIDTH))
RESIZE_HEIGHT = HEIGHT
NUM_CLASSES = 11

VERIFY_GRAPH = False
WRITE_LOG = False
WRITE_OUTPUT = False

def load_graph(frozen_graph_filename):
    # Load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph

def load_image(path):
    image = cv2.cvtColor(cv2.imread(path,-1), cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(image, (WIDTH, HEIGHT))
    return resized_image

if __name__ == '__main__':

    # Load graph
    graph = load_graph(GRAPH_NAME)

    if VERIFY_GRAPH:
        f = open("uv_graph_2.txt", 'w')
        # We can verify that we can access the list of operations in the graph
        for op in graph.get_operations():
            print(op.name)
            graph_shape = op.name + "\n"
            f.write(graph_shape)
        f.close()

    # Access the input and output nodes
    x = graph.get_tensor_by_name('prefix/Placeholder:0')
    y = graph.get_tensor_by_name('prefix/logits/Conv2D:0')

    # Launch a Session
    with tf.Session(graph=graph) as sess:

        count = START_COUNT

        f = open('./Test/predicting_time_frozen_180823.txt', 'w')

        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        for test_image in glob.glob(TEST_IMAGE_PATH + '/*.png'):
            print("{} th image detecting ... ".format(count))
            loaded_image = load_image(test_image)
            input_image = np.expand_dims(np.float32(loaded_image[:HEIGHT, :WIDTH]), axis=0) / 255.0

            st = time.time()

            #if WRITE_LOG:
            #    writer = tf.summary.FileWriter('./Test/step', sess.graph)

            output_image = sess.run(y, feed_dict={x: input_image}, options=options, run_metadata=run_metadata)
            #output_image = sess.run(y, feed_dict={x: input_image})
            run_time = time.time() - st

            if WRITE_OUTPUT:
                tmp_f = open("output.txt", 'w')
                tmp_f.write("==== output_image : " + "\n" + "{}".format(output_image))

            if WRITE_LOG:
                # Create the Timeline object, and write it to a json file
                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                with open('./checkpoints/timeline_180830.json', 'w') as t:
                    t.write(chrome_trace)

            output_image = np.array(output_image[0, :, :, :])

            if WRITE_OUTPUT:
                tmp_f.write(
                    "\n\n==== output_image_to_np_array : " + "\n" + "{}".format(output_image))

            output_image = helpers.reverse_one_hot(output_image)

            if WRITE_OUTPUT:
                tmp_f.write("\n\n===== output_image_to_reverse_one_hot : " + "\n" + "{}".format(output_image))

            # this needs to get generalized
            class_names_list, label_values = helpers.get_label_info(os.path.join(DATASET_NAME, LABEL_NAME))

            out_vis_image = helpers.colour_code_segmentation(output_image, label_values)
            if WRITE_OUTPUT:
                tmp_f.write("\n\n===== color code : " + "\n" + "{}".format(out_vis_image))
                tmp_f.close()

            file_name = utils.filepath_to_name(test_image)

            resized_out_vis_image = cv2.resize(np.uint8(out_vis_image), (WIDTH, RESIZE_HEIGHT))
            cv2.imwrite("%s/%s_pred.png" % (RESULT_PATH, file_name), cv2.cvtColor(resized_out_vis_image, cv2.COLOR_RGB2BGR))

            print("")
            print("Finished!")
            print("Wrote image " + "%s/%s_pred.png" % ("Test", file_name))
            print(run_time)

            count = count + 1
            predicting_time = "{}_pred.png".format(file_name) + ", " + "{} seconds".format(run_time) + "\n"

            f.write(predicting_time)

        f.close()

