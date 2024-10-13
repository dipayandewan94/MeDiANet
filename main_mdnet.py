if __name__ == "__main__":
    import argparse
    import sys
    import os


    sys.path.append(os.path.join(os.path.dirname(__file__), 'tensorflow_mdnet'))
    sys.path.append(os.path.join(os.path.dirname(__file__), 'pytorch_mdnet'))

    # import train_69
    # import train_117

    parser = argparse.ArgumentParser(description="Train a model using PyTorch or TensorFlow with different model types")

    parser.add_argument('--framework', choices=['pytorch', 'tensorflow'], required=True,
                        help='Choose the framework to train the model: pytorch or tensorflow')

    parser.add_argument('--model', choices=['117', '69'], required=True,
                        help='Choose the model version to train: 117 or 69')

    parser.add_argument('--eval', choices=['117', '69'], required=False,
                        help='Choose the model version to evaluate: 117 or 69')

    args = parser.parse_args()


    # Calling trainer for PyTorch Models:
    if args.framework == 'pytorch':
        import train_69
        import train_117
        
        if args.model == '117':
            train_117.train()  
        else:
            train_69.train() 


    # Calling trainer for TensorFlow Models:
    if args.framework == 'tensorflow':
        import train_69tf
        import train_117tf
        
        if args.model == '117':
            if args.eval == '117':
                eval_69tf.eval()
            else:
                train_117tf.train()  
        else:
            if args.eval == '69':
                eval_69tf.eval()
            else:
                train_69tf.train() 