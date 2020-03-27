from keras.models import load_model

def main(model, outpath):
    model = load_model(model)
    print(model.outputs)
    print(model.inputs)


if __name__ == '__main__':

    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Convert Keras Models to Tensorflow')
    parser.add_argument('--model', required=False,
                        metavar="/path/to/h5/model/",
                        help='Directory of the Balloon dataset', 
                        default="/datasets/models/mrcnn/fullstack_fullnetwork/chips20200326T0339/mask_rcnn_chips_0030.h5")
                        
    parser.add_argument('--outpath', required=False,
                        metavar="/path/to/output",
                        help="output path of the .pb file'", 
                        default="./output")
    args = parser.parse_args()

    main(args.model, args.outpath)