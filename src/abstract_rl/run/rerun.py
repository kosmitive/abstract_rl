from os.path import join

import numpy as np
from scandir import scandir


def rerun(alg_fn, root_folder='data'):

    # first select available environment
    print("Please select environment:")
    envfolders = [f.name for f in scandir(root_folder) if f.is_dir()]
    env_opt = choose_option(envfolders)
    line_segs = 30
    print("-" * line_segs)

    print("Please select algorithm:")
    env_root = join(root_folder, envfolders[env_opt])
    algfolders = [f.name for f in scandir(env_root) if f.is_dir()]
    alg_opt = choose_option(algfolders)
    print("-" * line_segs)

    print("Please select run:")
    alg_root = join(env_root, algfolders[alg_opt])
    run_folders = [f.name for f in scandir(alg_root) if f.is_dir()]
    run_folders.sort(reverse=True)
    run_opt = choose_option(run_folders)
    print("-" * line_segs)

    # obtain the root
    run_root = join(alg_root, run_folders[run_opt])
    run_config = {

        # simulations settings
        'reload': True,
        'reload_dir': run_root,
    }
    alg_fn(run_config)


def choose_option(options):

    # display in chunks of 10
    line_segs = 30
    print("-" * line_segs)
    chunk_i = 0
    lo = len(options)
    res = -1
    chunk_size = 10

    while chunk_i < lo:

        # display one batch
        up_chunk = np.minimum(chunk_i + chunk_size, lo)
        for k in range(chunk_i, up_chunk):
            print(f"[{k-chunk_i}] {options[k]}")

        if up_chunk != lo: print(f"[{up_chunk}] continue ...")
        print("-" * line_segs)
        t_res = input("Which option to choose? ")
        t_res = int(t_res)
        if t_res < 0:
            print("-" * line_segs)
            print("Must be bigger or equal to 0.")
            print("-" * line_segs)
        elif t_res > up_chunk:

            print("-" * line_segs)
            print(f"Must be smaller or equal to {up_chunk}.")
            print("-" * line_segs)

        elif t_res < up_chunk:
            return t_res + chunk_i

        else:
            if up_chunk == lo:
                print("-" * line_segs)
                print(f"Nothing to continue here.")
                print("-" * line_segs)

            else:
                chunk_i = up_chunk

    return res
