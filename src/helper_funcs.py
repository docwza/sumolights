import os, datetime

def check_and_make_dir(path):
    if not os.path.isdir(path):
        try:
            os.makedirs(path)
        except OSError:
            print ("Creation of the directory "+str(path)+" failed")

def write_lines_to_file(fp, write_type, lines):
    with open(fp, write_type) as f:
        f.writelines([l+'\n' for l in lines])

def write_line_to_file(fp, write_type, line):
    with open(fp, write_type) as f:
        f.write(line+'\n')

def get_time_now():
    now = datetime.datetime.now()
    now = str(now).replace(" ","-")
    now = now.replace(":","-")
    return now

def write_to_log(s):
    fp = 'tmp/'
    check_and_make_dir(fp)
    fp += 'log.txt'
    t = get_time_now()
    write_line_to_file(fp, 'a+', t+':: '+s)
