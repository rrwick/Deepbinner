"""
Copyright 2018 Ryan Wick (rrwick@gmail.com)
https://github.com/rrwick/Deepbinner/

This file is part of Deepbinner. Deepbinner is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version. Deepbinner is distributed
in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
details. You should have received a copy of the GNU General Public License along with Deepbinner.
If not, see <http://www.gnu.org/licenses/>.
"""


def realtime(args):
    pass



#
#
# def make_output_dirs(out_dir, classifications):
#     if pathlib.Path(out_dir).is_file():
#         sys.exit('Error: {} is an existing file'.format(out_dir))
#     print('Making output directories:')
#     try:
#         if not pathlib.Path(out_dir).is_dir():
#             os.makedirs(out_dir, exist_ok=True)
#             print('  {}/'.format(out_dir))
#         class_dirs = sorted(class_to_barcode_dir(x) for x in set(classifications.values()))
#         for class_dir in class_dirs:
#             d = pathlib.Path(out_dir) / class_dir
#             if d.is_dir():
#                 sys.exit('Error: {} already exists'.format(d))
#             os.makedirs(str(d), exist_ok=True)
#             print('    {}/'.format(d))
#     except (FileNotFoundError, OSError, PermissionError):
#         sys.exit('Error: unable to create output directory {}'.format(out_dir))
#     print()
#
#
# def class_to_barcode_dir(classification):
#     if classification is None:
#         return 'unclassified'
#     else:
#         return 'barcode{:02d}'.format(classification)
