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


native_start_kit_adapter = 'AATGTACTTCGTTCAGTTACGTATTGCT'
native_start_barcodes = {'01': 'AAGGTTAACACAAAGACACCGACAACTTTCTTCAGCACCT',
                         '02': 'AAGGTTAAACAGACGACTACAAACGGAATCGACAGCACCT',
                         '03': 'AAGGTTAACCTGGTAACTGGGACACAAGACTCCAGCACCT',
                         '04': 'AAGGTTAATAGGGAAACACGATAGAATCCGAACAGCACCT',
                         '05': 'AAGGTTAAAAGGTTACACAAACCCTGGACAAGCAGCACCT',
                         '06': 'AAGGTTAAGACTACTTTCTGCCTTTGCGAGAACAGCACCT',
                         '07': 'AAGGTTAAAAGGATTCATTCCCACGGTAACACCAGCACCT',
                         '08': 'AAGGTTAAACGTAACTTGGTTTGTTCCCTGAACAGCACCT',
                         '09': 'AAGGTTAAAACCAAGACTCGCTGTGCCTAGTTCAGCACCT',
                         '10': 'AAGGTTAAGAGAGGACAAAGGTTTCAACGCTTCAGCACCT',
                         '11': 'AAGGTTAATCCATTCCCTCCGATAGATGAAACCAGCACCT',
                         '12': 'AAGGTTAATCCGATTCTGCTTCTTTCTACCTGCAGCACCT'}


native_end_kit_adapter = 'AGCAATACGTAACTGAACGAAGT'
native_end_barcodes = {'01': 'AGGTGCTGAAGAAAGTTGTCGGTGTCTTTGTGTTAACCTT',
                       '02': 'AGGTGCTGTCGATTCCGTTTGTAGTCGTCTGTTTAACCTT',
                       '03': 'AGGTGCTGGAGTCTTGTGTCCCAGTTACCAGGTTAACCTT',
                       '04': 'AGGTGCTGTTCGGATTCTATCGTGTTTCCCTATTAACCTT',
                       '05': 'AGGTGCTGCTTGTCCAGGGTTTGTGTAACCTTTTAACCTT',
                       '06': 'AGGTGCTGTTCTCGCAAAGGCAGAAAGTAGTCTTAACCTT',
                       '07': 'AGGTGCTGGTGTTACCGTGGGAATGAATCCTTTTAACCTT',
                       '08': 'AGGTGCTGTTCAGGGAACAAACCAAGTTACGTTTAACCTT',
                       '09': 'AGGTGCTGAACTAGGCACAGCGAGTCTTGGTTTTAACCTT',
                       '10': 'AGGTGCTGAAGCGTTGAAACCTTTGTCCTCTCTTAACCTT',
                       '11': 'AGGTGCTGGTTTCATCTATCGGAGGGAATGGATTAACCTT',
                       '12': 'AGGTGCTGCAGGTAGAAAGAAGCAGAATCGGATTAACCTT'}


rapid_start_kit_adapter = 'CGTTCAGTTACGTATTGCTGTTTTCGCATTTATCGTGAA'
rapid_start_barcodes = {'01': 'TATTGCTCACAAAGACACCGACAACTTTCTTGTTTTCGC',
                        '02': 'TATTGCTACAGACGACTACAAACGGAATCGAGTTTTCGC',
                        '03': 'TATTGCTCCTGGTAACTGGGACACAAGACTCGTTTTCGC',
                        '04': 'TATTGCTTAGGGAAACACGATAGAATCCGAAGTTTTCGC',
                        '05': 'TATTGCTAAGGTTACACAAACCCTGGACAAGGTTTTCGC',
                        '06': 'TATTGCTGACTACTTTCTGCCTTTGCGAGAAGTTTTCGC',
                        '07': 'TATTGCTAAGGATTCATTCCCACGGTAACACGTTTTCGC',
                        '08': 'TATTGCTACGTAACTTGGTTTGTTCCCTGAAGTTTTCGC',
                        '09': 'TATTGCTAACCAAGACTCGCTGTGCCTAGTTGTTTTCGC',
                        '10': 'TATTGCTGAGAGGACAAAGGTTTCAACGCTTGTTTTCGC',
                        '11': 'TATTGCTTCCATTCCCTCCGATAGATGAAACGTTTTCGC',
                        '12': 'TATTGCTTCCGATTCTGCTTCTTTCTACCTGGTTTTCGC'}
