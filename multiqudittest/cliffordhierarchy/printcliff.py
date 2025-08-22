import numpy as np;
import os
import argparse

# Main program
if __name__ == "__main__":
    file_name = 'n2-c2-gates-d$PRIME-asGATES.npy'
    parser = argparse.ArgumentParser(description=f"Gets the specified Clifford gate by index from file {file_name}")
    parser.add_argument("d", type=int, help="Index")
    parser.add_argument("index", type=int, help="Index")
    args = parser.parse_args()
    if args.index < 0:
        raise argparse.ArgumentTypeError("index must be non-negative")
    if args.d < 0:
        raise argparse.ArgumentTypeError("d must be non-negative")
    
    file_name = f'n2-c2-gates-d{args.d}-asGATES.npy'
    file_name_compresed = f'n2-c2-gates-d{args.d}-asGATES.npz'

    if os.path.exists(file_name_compresed):
        print(f"Loaded compressed gates")
        loaded_data = np.load(file_name_compresed)
        C2 = loaded_data['data']
    elif os.path.exists(file_name):
        with open(file_name, 'rb') as f:
            C2 = np.load(f, allow_pickle=True)
    else:
        print(f"Missing {file_name} or .npz")
        exit(1)

    print(f"Loaded {len(C2)} gates. Shape is {C2.shape}")

    if not os.path.exists(file_name_compresed):
        print(f"Saving as compressed file {file_name_compresed}")
        np.savez_compressed(file_name_compresed, data=C2)
        print(f"Sone saving as compressed")

    print(f"Printing out the {args.index}th gate (shape)")
    print(f"Shape {C2[args.index].shape}")
    print(C2[args.index])
