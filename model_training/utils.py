import os
import cv2
import numpy as np

from google.protobuf import text_format

from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import importer
from tensorflow.python.platform import app  # noqa
from tensorflow.python.platform import gfile
from tensorflow.python.training import saver as saver_lib


def labels_on_images(images, groundtruth_label, predicted_label):
    """
    Burn labels/lotwafer info on images (left corner)
    :param images: Unmarked images
    :param groundtruth_label: ground truth label
    :param predicted_label: predicted label
    :param lotids: lotid list
    :param waferids: waferid list
    :return: marked images
    """

    assert len(images) == len(groundtruth_label) == len(predicted_label), "Image length: {0}, " \
                                                                          "Ground Truth length: {1} " \
                                                                          "and Predicted Label" \
                                                                          " length {2} dont " \
                                                                          "match".format(len(images),
                                                                                         len(groundtruth_label),
                                                                                         len(predicted_label))
    image_burned_stack = []
    for idx in range(len(images)):
        # default, unburned image
        image_unburned = images[idx]
        # rescale images back to 0-255
        image_unburned = np.multiply(image_unburned, 255)
        # covert the image back to a uint8 range
        image_unburned = image_unburned.astype(np.uint8)
        # rounded bin counts
        predicted_round = round(predicted_label[idx], 2)
        groundtruth_round = round(groundtruth_label[idx], 2)
        # prepare the note to burn in the image
        note = "Pred:" + str(predicted_round) + " Act:" + str(groundtruth_round)

        # determine the color
        if predicted_round == groundtruth_round:
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)
        # burn the text on the image
        image_burned = cv2.putText(image_unburned, str(note), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 1)

        # add to a stack
        image_burned_stack.append(image_burned)

    return image_burned_stack


def freeze_graph(input_graph,
                 input_saver,
                 input_binary,
                 input_checkpoint,
                 output_node_names,
                 restore_op_name,
                 filename_tensor_name,
                 output_graph,
                 clear_devices,
                 initializer_nodes,
                 variable_names_blacklist=""):

    """Converts all variables in a graph and checkpoint into constants."""

    if not gfile.Exists(input_graph):
        print("Input graph file '" + input_graph + "' does not exist!")
        return -1

    if input_saver and not gfile.Exists(input_saver):
        print("Input saver file '" + input_saver + "' does not exist!")
        return -1

    # 'input_checkpoint' may be a prefix if we're using Saver V2 format
    if not saver_lib.checkpoint_exists(input_checkpoint):
        print("Input checkpoint '" + input_checkpoint + "' doesn't exist!")
        return -1

    if not output_node_names:
        print("You need to supply the name of a node to --output_node_names.")
        return -1

    input_graph_def = graph_pb2.GraphDef()
    mode = "rb" if input_binary else "r"
    with gfile.FastGFile(input_graph, mode) as f:
        if input_binary:
            input_graph_def.ParseFromString(f.read())
        else:
            text_format.Merge(f.read(), input_graph_def)
    # Remove all the explicit device specifications for this node. This helps to
    # make the graph more portable.
    if clear_devices:
        for node in input_graph_def.node:
            node.device = ""
    _ = importer.import_graph_def(input_graph_def, name="")

    with session.Session() as sess:
        if input_saver:
            with gfile.FastGFile(input_saver, mode) as f:
                saver_def = saver_pb2.SaverDef()
                if input_binary:
                    saver_def.ParseFromString(f.read())
                else:
                    text_format.Merge(f.read(), saver_def)
                saver = saver_lib.Saver(saver_def=saver_def)
                saver.restore(sess, input_checkpoint)
        else:
            sess.run([restore_op_name], {filename_tensor_name: input_checkpoint})
            if initializer_nodes:
                sess.run(initializer_nodes)

        variable_names_blacklist = (variable_names_blacklist.split(",") if
                                    variable_names_blacklist else None)
        output_graph_def = graph_util.convert_variables_to_constants(
            sess,
            input_graph_def,
            output_node_names.split(","),
            variable_names_blacklist=variable_names_blacklist)

    with gfile.GFile(output_graph, "wb") as f:
        f.write(output_graph_def.SerializeToString())
    print("%d ops in the final graph." % len(output_graph_def.node))


def create_freeze_graph(global_step, logdir, logdir_train, epochs, model_type, dropout_rate,
                        input_graph='output_graph.pb'):

    op_tensor = 'tower_0/InceptionV3/Predictions/Reshape_1'

    # generate a frozen graph at the end of a training session
    # fetch the input checkpoint file name
    with open(logdir_train + '/checkpoint', 'r') as checkpoint:
        for ckptline in checkpoint.readlines():
            if 'model_checkpoint_path' in ckptline:
                input_checkpoint = ckptline.split(": ")[1].replace('"', '').strip()
                break

    # did frozen graph must contain meta data within its name along with its associated param,
    # params are seperated with a _ and values are sep with a -. For example, Iteration-'value'_part-'value'-
    output_frozen_graph_name = "ITERATION-{0}_EPOCHS-{1}_NETWORK-{2}_DROPOUT-{3}.freeze.pb".format(
        global_step,
        epochs,
        model_type,
        int(dropout_rate * 100))

    freeze_graph(input_graph=os.path.join(logdir, input_graph),
                 input_saver="",
                 input_binary=False,
                 input_checkpoint=input_checkpoint,
                 output_node_names=op_tensor,
                 restore_op_name="save/restore_all",
                 filename_tensor_name="save/Const:0",
                 output_graph=os.path.join(logdir,
                                           output_frozen_graph_name),
                 clear_devices=True,
                 initializer_nodes='init_1',
                 variable_names_blacklist="")

    return 0