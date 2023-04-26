import argparse
import os
import sys
import pathlib
import shutil
import math

import subprocess
from datetime import datetime
import numpy as np
import codecs
import uuid 
from varname.helpers import debug
import random
from math import cos, sin, fabs, pi, asin, sqrt
import platform 

def curDateTime():
    now = datetime.now()
    # return now.strftime("%Y_%m.%d.%H.%M.%S.%f")
    return now.strftime("%Y.%m.%d.%H.%M.%S")

def ls(src, sort=True):
    """os.listdir modify with check os.path.isdir(src)
    
    Arguments:
        src {[str]} -- [dir path]
    
    Returns:
        [False] -- [if src is not dir]
    """
    if os.path.isdir(src):
        if sort:
            return sorted(os.listdir(src))
        else:
            return os.listdir(src)
    raise NotADirectoryError('not a dir', src)

def ls_wc(src):
    return sum([len(files) for r, d, files in os.walk(src)])

def rmdir(dir_p):
    shutil.rmtree(dir_p)

def rmrf(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.remove(path)

def split_path(path):
    return os.path.normpath(path).split(os.path.sep)

def isiter(obj):
    try:
        _ = (e for e in obj)
    except TypeError:
        return False
    return True

def flat_list(l=[]):
    return [item for sublist in l for item in sublist]

def find_files(dir, search_substr, only_up_dir=False, abs_p=False):
    found_files = []
    for obj_name in ls(dir):
        obj_path = os.path.join(dir, obj_name)
        if os.path.isdir(obj_path):
            found_files += find_files(obj_path, search_substr, only_up_dir)
        if search_substr in obj_name or search_substr is None:
            if only_up_dir:
                found_files.append(os.path.join(dir.split(os.sep)[-1], obj_name))
            else:
                if abs_p:
                    found_files.append(os.path.abspath(obj_path))
                else:
                    found_files.append(obj_path)
    return found_files
    
def cp_r(src, dst):
    if os.path.isfile(src):
        ensure_folder(os.path.split(dst)[0])
        shutil.copy(src, dst)
        return True
    if os.path.isdir(src):
        ensure_folder(dst)
        for item in ls(src):
            item_p = os.path.join(src, item)
            if os.path.isfile(item_p):
                dst_item_p = os.path.join(dst, item) 
                shutil.copy(item_p, dst_item_p)
            elif os.path.isdir(item_p):
                new_dst = os.path.join(dst, item)
                ensure_folder(new_dst)
                cp_r(item_p, new_dst)
    return False

def ensure_folder(dir_fname):
    if not os.path.exists(dir_fname):
        try:
            pathlib.Path(dir_fname).mkdir(parents=True, exist_ok=True)
        except PermissionError:
            print('Unable to create {} directory. Permission denied'.format(dir_fname))
            return False
    return True

def boolean_string(s):
    if isinstance(s, bool):
        return s
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string, use False or True')
    return s == 'True'

def shell(command_sh):
    try:
        process = subprocess.Popen(command_sh.split(), stdout=subprocess.PIPE)
    except Exception as e:
        print('shell:: subprocess.Popen', command_sh, e)
    try:
        output, error = process.communicate()
        return output, error
    except Exception as e:
        print('shell:: subprocess.communicate()', command_sh, e)
    return None

def isDir(dir_path):
    if not os.path.isdir(dir_path):
        raise NotADirectoryError(dir_path)
    
def isFile(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(file_path)

def argparserIO():
    parser = argparse.ArgumentParser(
                                    )
    parser.add_argument(
                        '--path', type=str,
                        )
    parser.add_argument(
                        '--outpath', type=str,
                        )
    parser.add_argument(
                        '--netname', type=str, default=''
                        )
    parser.add_argument(
                        '--params_dir', type=str, default=''
                        )
    parser.add_argument(
                        '--epoch', type=int, default=0
                        )
    return parser

def hex2int(h):
    return int(h, 16)

def hex2utf(s):
    return codecs.decode(s, 'hex').decode('utf-8')

def hex2ru(s):
    return s.encode().decode("cp1251").encode("utf-8")

def str2byte(s) -> np.array:
    return np.array([x for x in s.encode('utf-8')])

def str2byte_norm(s) -> np.array:
    return np.array([x/255.0 for x in s.encode('utf-8')])

def path2list(path):
    return path.split(os.sep)

def euclidean_dist(point_a, point_b):
    return np.sqrt(np.sum(np.square(point_a - point_b)))
    
def token():
    return uuid.uuid4().hex

def cp_all(inp_dir, substrs, outdir):
    ensure_folder(outdir)
    was = ls_wc(outdir)
    print('cp_all::processing dir {}'.format(inp_dir))
    for substr in substrs:
        fs = find_files(inp_dir, substr)
        for f in fs:
            ext = os.path.splitext(f)[1]
            outname = '{}{}'.format(token(), ext)
            out_p = os.path.join(outdir, outname)
            cp_r(f, out_p)
    print('dir {} ({}) done with {} new images'.format(outdir, ls_wc(outdir), ls_wc(outdir)-was))
    #cp_all('/home/ilpech/datasets', ['.png','.jpg'], '/home/ilpech/datasets/out')

def readSearchXlsxReport(file_path, sheet_name=''):
    file_abs_path = os.path.abspath(file_path)
    print('readSearchGUIReport::reading data at::', file_abs_path)
    df_sheet_all = pd.read_excel(file_path, sheet_name=None, engine='openpyxl')
    if len(sheet_name) != 0:
        try:
            return df_sheet_all[sheet_name]
        except KeyError as e:
            print(e)
            print('readSearchGUIReport::sheets in file::')
            [print(x) for x in df_sheet_all]
            raise Exception(
                'readSearchGUIReport::selected sheet {} is not accessible for file {}'.format(
                    sheet_name, file_path
                )
            )
    selected_sheet_name = list(df_sheet_all.keys())[0]
    if len(df_sheet_all) > 1:
        selected_sheet_name = [x for x in df_sheet_all if 'Total_Report' in x]
        if len(selected_sheet_name) == 0:
            raise Exception('more than one sheet and no one with Total_Report')
        selected_sheet_name = selected_sheet_name[0]
    print('readSearchGUIReport::selected_sheet_name::', selected_sheet_name)
    return df_sheet_all[selected_sheet_name]

def extract_nbr(input_str, tp):
    if input_str is None or input_str == '':
        return 0
    out_number = ''
    for ele in input_str:
        if ele.isdigit() or ele=='.':
            out_number += ele
    return tp(out_number) 

def get_key_from_str(str_p, key, delim, templ=None, rename=None):
    spl = str_p.split(delim)
    if rename is None:
        rename = key
    d = [(rename, spl[x+1]) for x in range(len(spl)) if key in spl[x]]
    if not len(d):
        return None
    d = d[0]
    if templ is not None:
        return (d[0], extract_nbr(d[1], templ))
    if d == 'None':
        return None
    return d

def shuffle_dict(d):
    l = list(d.items())
    random.shuffle(l)
    return dict(l)

def listfind(set_, subset_):
    '''
    returns found (ids_of_subset_found_objects_in_set, found_values)
    '''
    return (
        [j for j in range(len(set_)) if set_[j] in subset_],
        [x for x in set_ if x in subset_]
    )
    
def list2nonEmptyIds(l):
    return [j for j in range(len(l)) if l[j]]

def roundUp(a):
    '''
    округление всегда в большую сторону
    '''
    return np.ceil(a)

def setindxs(l_, indxs):
    return [l_[indxs[i]] for i in range(len(indxs))]
    
def shuffle(l):
    '''
    shuffle list saving order of replaced indexes
    
    returns (shuffled_list, order_of_shuffled_indexes)
    '''
    indxs = [i for i in range(len(l))]
    np.random.shuffle(indxs)
    return setindxs(l, indxs), indxs

def is_number(s):
    try:
        float(s)
        if not math.isnan(s):
            return True
        return False
    except ValueError:
        return False

def denorm_shifted_log(data):
    if is_number(data):
        # if data <= 0:
        #     return data
        return np.exp(float(data))-1
    raise ValueError("denorm_shifted_log::not a number {}".format(data))

def norm_shifted_log(data, c=1):
    if is_number(data):
        data = float(data)
        if data <= -c:
            print(f'norm_shifted_log::invalid value {data}')
            return None
        return np.log(data + c)
    raise ValueError("norm_shifted_log::not a number {}".format(data))

def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string, use False or True')
    return s == 'True'

def str2vec(str_s):
    v_start = [i for i in range(len(str_s)) if str_s[i] == '['][0]
    v_fin = [i for i in range(len(str_s)) if str_s[i] == ']'][0]
    v_str = str_s[v_start+1:v_fin]
    v_str = v_str.replace(' ', '')
    return np.fromstring(v_str, sep=',')

class BatchFullException(Exception):
    pass

def ls_dir(path):
    return [x for x in ls(path) if os.path.isdir(os.path.join(path, x))]

def format_floats_list(l):
    out = '[ '
    for i, e in enumerate(l):
        if i != len(l)-1:
            out += '{:.8f},'.format(e)
        else:
            out += '{:.8f} ]'.format(e)
    return out

def examine_dir(inp_path):
    dataset_classes = [x for x in ls(inp_path) if os.path.isdir(os.path.join(inp_path, x))]
    dataset_classes_cnts = []
    for d_cls in dataset_classes:
        d_path = os.path.join(inp_path, d_cls)
        dataset_classes_cnts.append(ls_wc(d_path))
    dataset_classes_cnts = np.array(dataset_classes_cnts)
    dirs_cnts = []
    for i in range(len(dataset_classes)):
        dirs_cnts.append((dataset_classes[i], dataset_classes_cnts[i]))
    return dirs_cnts

def metric2plot(
    x, 
    ys, 
    y_labels, 
    x_label, 
    wintitle, 
    colors, 
    linewidths, 
    out_path, 
    start_from=0
):
    import pylab
    pylab.figure(figsize=(35, 6), dpi=100)
    if start_from:
        x = x[start_from:]
        ys = [y[start_from:] for y in ys]
    for i in range(len(ys)):
        pylab.plot(
            x, 
            ys[i], 
            color=colors[i], 
            linewidth=linewidths[i], 
            label=y_labels[i]
        )
    pylab.title(wintitle)
    pylab.xlabel(x_label)
    pylab.legend(loc='upper right')
    pylab.grid()
    new_path = os.path.join(out_path, wintitle)
    if '.png' not in new_path:
        new_path = new_path + '.png'
    pylab.savefig(new_path, dpi=100)
    print('saved at', new_path)
    pylab.close()

def epoch2datetime(ts):
    return datetime.fromtimestamp(float(ts)/1000.)

def epoch2date(ts):
    return epoch2datetime(ts).date()
    
def img2info(img):
    return 'img:{} max:{} min:{} mean:{}'.format(
        img.shape,
        np.max(img),
        np.min(img),
        np.mean(img),
    )

class PointEN:
    def __init__(self, lat, long, alt=0):
        self.lat = lat
        self.long = long
        self.alt = alt

    def coords(self):
        return (self.long, self.lat, self.alt)

def scale(val, src, dst):
    """
    scale the given value from the scale of src to the scale of dst.
    """
    return ((val - src[0]) / (src[1]-src[0])) * (dst[1]-dst[0]) + dst[0]

def convert_bytes(num):
    """
    this function will convert bytes to MB.... GB... etc
    """
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if num < 1024.0:
            return "%3.1f %s" % (num, x)
        num /= 1024.0

def file_size(file_path):
    """
    this function will return the file size
    """
    if os.path.isfile(file_path):
        file_info = os.stat(file_path)
        return convert_bytes(file_info.st_size)

def isUbuntu():
    return 'ubuntu' in platform.version().lower()

def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def list2chunks(l, n):
    return list(divide_chunks(l,n))

def prime_factorization(n):
    """Return the prime factorization of int value.

    Parameters
    ----------
    n : int
        The number for which the prime factorization should be computed.

    Returns
    -------
    dict[int, int]
        List of tuples containing the prime factors and multiplicities of `n`.

    """
    prime_factors = {}
    i = 2
    while i**2 <= n:
        if n % i:
            i += 1
        else:
            n /= i
            try:
                prime_factors[i] += 1
            except KeyError:
                prime_factors[i] = 1
    if n > 1:
        try:
            prime_factors[n] += 1
        except KeyError:
            prime_factors[n] = 1
    return prime_factors

def findkey(obj, key):
    if key in obj: return obj[key]
    for k, v in obj.items():
        if isinstance(v,dict):
            item = findkey(v, key)
            if item is not None:
                return item