from argparse import ArgumentParser
import os
import pandas as pd
import csv
import zipfile
import re

# command line arguments parser
parser = ArgumentParser()
parser.add_argument(dest='type', type=str, default='all', help="Which data type to convert? [all|tweets|chess|music|shakespeare|javascript|typescript|json|html]")
parser.add_argument('--data-dir', '-d', dest='data_dir', type=str, default='../datasets/', help="Path to the directory containing the raw data.")
parser.add_argument('--output-dir', '-o', dest='output_dir', type=str, default='./data/', help="Path to the output directory.")
parser.add_argument('--short-filename', '-s', dest='short_filename', type=str, default='false', help="Does not include parameter info in filename.")
parser.add_argument('--postfix', '-p', dest='postfix', type=str, default='', help="Postfix is appended to the filename stem before the suffix and parameter info (if applicable).")
parser.add_argument('--num-samples', '-n', dest='num_samples', type=int, default=1000, help="Max number of samples to be exported.")
parser.add_argument('--max-length', dest='max_length', type=int, default=2000, help="Max length of a sample.")
parser.add_argument('--min-length', dest='min_length', type=int, default=10, help="Min length of a sample.")
parser.add_argument('--preserve-lines', dest='preserve_lines', type=str, default='false', help="Preserve line breaks in data and don't collapse the whole sample into a single line (except html).")
parser.add_argument('--preserve-form', dest='preserve_form', type=str, default='false', help="Preserve original form including linebreaks and comments (javascript and typescript only), and urls (tweets only)")
args = parser.parse_args()

# form requires newlines to be preserved
if args.preserve_form == 'true':
    args.preserve_lines = 'true'

# collapsing sample into one line requires form not to be preserved
if args.preserve_lines == 'false':
    args.preserve_form = 'false'

# set postfix for output files if short-filename is false
if args.postfix != '':
    args.postfix = '_' + args.postfix
if args.short_filename == 'false':
    args.postfix += f'_n{args.num_samples}_min{args.min_length}_max{args.max_length}'
    if args.preserve_lines == 'false':
        args.postfix += '_nolines'
    else:
        args.postfix += '_lines'
    if args.preserve_form == 'false':
        args.postfix += '_noform'
    else:
        args.postfix += '_form'

# print arguments to show values in use
print(args)

# helper to use code samples in zip file
def process_zip(name, regs, args):
    with open(os.path.join(args.output_dir, name + args.postfix + '.txt'), 'w+') as fh:
        with zipfile.ZipFile(os.path.join(args.data_dir, name + '.zip'), 'r') as z:
            cnt = 0
            for entry in z.namelist():
                text = z.read(entry).decode('utf-8')
                for reg, sub in regs.items():
                    text = re.sub(reg, sub, text, flags=re.DOTALL)
                if len(text) > args.min_length and len(text) <= args.max_length:
                    sample = text.strip() + "\n"
                    if args.preserve_form == 'true':
                        sample += "\n\n"
                    fh.write(sample)
                    cnt += 1
                if cnt >= args.num_samples:
                    break


# dataset from: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi%3A10.7910%2FDVN%2FKJEBIL
if args.type in ['all','tweets']: # parse trump tweets
    print('prepare tweet data set...')
    df1 = pd.read_json(os.path.join(args.data_dir, 'realdonaldtrump-1.ndjson'), lines=True)
    df2 = pd.read_json(os.path.join(args.data_dir, 'realdonaldtrump-2.ndjson'), lines=True)
    df = pd.concat([df1, df2], sort=True)
    if args.preserve_lines == 'false':
        df.text = df.text.str.replace("\n"," ")
    if args.preserve_form == 'false':
        df.text = df.text.str.replace(r"https?://[^\s]+","")
    df['length'] = df.text.apply(len)
    filter = (df.text>'2017')&(df.text.str.startswith('RT')==False)&(df.length>args.min_length)
    df = df[filter]
    df.sample(args.num_samples).text.to_csv(os.path.join(args.output_dir, 'tweets' + args.postfix + '.txt'), index=False, header=False, quoting=csv.QUOTE_NONE, escapechar="\\", sep="\\")
    print('preparing tweet data set done.')

# dataset from: https://www.ficsgames.org/download.html | year: 2019, month: whole year, type: Standard (average rating > 2000)
if args.type in ['all','chess']: # parse chess games
    print('prepare chess data set...')
    with open(os.path.join(args.output_dir, 'chess' + args.postfix + '.txt'),'w+') as fh:
        with open(os.path.join(args.data_dir, 'ficsgamesdb_2019_standard2000_nomovetimes_110541.pgn')) as fp:
           line = fp.readline()
           cnt = 0
           while line and cnt < args.num_samples:
               if line.startswith('1.'):
                   fh.write(line)
                   cnt += 1
               line = fp.readline()
    print('preparing chess data set done.')

# dataset from: https://www.kaggle.com/raj5287/abc-notation-of-tunes/version/3
if args.type in ['all','music']: # parse abc songs
    print('prepare music data set...')
    with open(os.path.join(args.output_dir, 'music' + args.postfix + '.txt'),'w+') as fh:
        with open(os.path.join(args.data_dir, 'abc_notation_songs.txt')) as fp:
            line = fp.readline()
            cnt = 0
            song = ""
            while line and cnt < args.num_samples:
                if len(line) < 2 or line[1:2] == ':':
                    if song != "":
                        fh.write(song + "\n")
                        cnt += 1
                        song = ""
                elif args.preserve_lines == 'false':
                    song += " " + line.strip()
                else:
                    fh.write(line.strip() + "\n")
                line = fp.readline()
    print('preparing music data set done.')

# dataset from: https://www.kaggle.com/kingburrito666/shakespeare-plays
if args.type in ['all','shakespeare']: # parse shakespeare plays
    print('prepare shakespeare data set...')
    df = pd.read_csv(os.path.join(args.data_dir, 'shakespeare_data.csv'))
    if args.preserve_lines == 'false':
        df = df[df.Player!=''].groupby(['Play','PlayerLinenumber'],as_index=False).agg(' '.join)
    df.sample(args.num_samples).PlayerLine.to_csv(os.path.join(args.output_dir, 'shakespeare' + args.postfix + '.txt'), index=False, header=False, quoting=csv.QUOTE_NONE, escapechar="\\", sep="\\")
    print('preparing shakespeare data set done.')

# dataset from: javascript files from https://www.sri.inf.ethz.ch/js150
if args.type in ['all','javascript']: # parse javascript files
    print('prepare javascript data set...')
    regexes = {}
    if args.preserve_form == 'false':
        regexes[r'(//[^\n]*)?\n|/\*.*?\*/'] = '\n'
        regexes[r'\n\s*\n'] = '\n'
    if args.preserve_lines == 'false':
        regexes[r'\s+'] = ' '
    process_zip('javascript', regexes, args)
    print('preparing javascript data set done.')

# dataset from: typescript files collected from standard angular app
if args.type in ['all','typescript']: # parse typescript files
    print('prepare typescript data set...')
    regexes = {}
    if args.preserve_form == 'false':
        regexes[r'(//[^\n]*)?\n|/\*.*?\*/'] = '\n'
        regexes[r'\n\s*\n'] = '\n'
    if args.preserve_lines == 'false':
        regexes[r'\s+'] = ' '
    process_zip('typescript', regexes, args)
    print('preparing typescript data set done.')

# dataset from: json files collected from standard angular app
if args.type in ['all','json']: # parse json files
    print('prepare json data set...')
    regexes = {}
    if args.preserve_lines == 'false':
        regexes[r'\s+'] = ' '
    process_zip('json', regexes, args)
    print('preparing json data set done.')

# dataset from: https://www.kaggle.com/zavadskyy/lots-of-code, https://gist.github.com/VladislavZavadskyy/e31ab07b03a5c22b11982c49669a400b
if args.type in ['all','html']: # parse html
    print('prepare html data set...')
    with open(os.path.join(args.output_dir, 'html' + args.postfix + '.txt'),'w+') as fh:
        with open(os.path.join(args.data_dir, 'html-dataset.txt')) as fp:
            data = fp.read()
            data = data.replace('<!DOCTYPE html>','\n<!DOCTYPE html>')
            lines = data.split('\n')
            cnt = 0
            sample = ""
            for line in lines:
                if line == "":
                    continue
                if sample != "" and line.startswith('<!DOCTYPE html>'):
                    fh.write(sample.strip() + "\n")
                    sample = ""
                    cnt += 1
                if cnt >= args.num_samples:
                    break
                line = re.sub(r'\s+', ' ', line)
                sample += line.strip() + " "
    print('preparing html data set done.')
