# -*- coding: utf-8 -*-
"""Find out if a legacy VTK file ends before the data that its headers declare.

VTK's legacy readers report a cut in some places and say nothing in others: a cut inside CELLS
gives a mesh with no elements, and a cut inside the last field gives wrong values. So this walks
the file by its declared counts. It needs no VTK.

It only reports a file that ends too early. A keyword or data type it does not know ends the
check and the file counts as complete, so a valid file is not rejected. The last word of a file
is a cut keyword only if a known keyword starts with it and nothing comes after it, not even
white space.
Some cuts can't be seen, because the rest of the file is then valid: one between two complete
sections, one right after the header of POINT_DATA or CELL_DATA, which VTK reads as no arrays, and
one inside the last value of an ASCII file. For the last, a warning is printed if the file has
no line break at its end, which VTK's writer and this package's writers always write.
"""
import re
import numpy as np
#%%
# Bytes per value in a binary file. VTK writes vtkidtype as 4 bytes, whatever the size of vtkIdType.
# A type that is missing here ends the check: 'string' has no fixed size, the size of 'long'
# depends on the program that wrote the file, and VTK writes too few bytes for a 'bit' array of
# several components.
BYTES_OF_VALUE_TYPE={b'char':1, b'unsigned_char':1, b'signed_char':1, b'short':2, b'unsigned_short':2,
                     b'int':4, b'unsigned_int':4, b'vtkidtype':4, b'float':4, b'double':8,
                     b'vtktypeint64':8, b'vtktypeuint64':8}
# Attribute keyword: values per node or per element. Its header is the keyword, a name and a type.
VALUES_OF_ATTRIBUTE={b'VECTORS':3, b'NORMALS':3, b'TENSORS':9, b'TENSORS6':6, b'GLOBAL_IDS':1, b'EDGE_FLAGS':1}
CELL_SECTION=(b'VERTICES', b'LINES', b'POLYGONS', b'TRIANGLE_STRIPS', b'CELLS')
# Every keyword that can start a section, including those that end the check.
KEYWORD=(b'DATASET', b'POINTS', *CELL_SECTION, b'CELL_TYPES', b'POINT_DATA', b'CELL_DATA', b'FIELD', b'SCALARS',
         *VALUES_OF_ATTRIBUTE, b'TEXTURE_COORDINATES', b'COLOR_SCALARS', b'LOOKUP_TABLE', b'PEDIGREE_IDS', b'METADATA',
         b'DIMENSIONS', b'ORIGIN', b'SPACING', b'ASPECT_RATIO', b'X_COORDINATES', b'Y_COORDINATES', b'Z_COORDINATES')
WORD=re.compile(rb'[^ \t\r\n\v\f]+')
#%%
class FileIsCut(Exception):
    pass

class UnknownContent(Exception):
    pass

class LegacyVtkFileCursor:
    #moves over the header words and the data blocks of a file, in ASCII or binary
    def __init__(self, text, position, is_binary):
        self.text=text
        self.position=position
        self.is_binary=is_binary
        if not is_binary:
            data=np.frombuffer(text, dtype=np.uint8)[position:]
            is_space=(data == 32) | ((data >= 9) & (data <= 13))
            is_space=np.concatenate(([True], is_space, [True]))
            #the changes between space and word are: start of word 0, end of word 0, start of word 1, ...
            change=np.flatnonzero(is_space[1:] != is_space[:-1])
            self.word_start=change[0::2]+position
            self.word_end=change[1::2]+position
            self.word=0

    def skip_space(self):
        while self.position < len(self.text) and self.text[self.position] in b' \t\r\n\v\f':
            self.position+=1

    def at_end(self):
        if not self.is_binary:
            return self.word >= len(self.word_start)
        self.skip_space()
        return self.position >= len(self.text)

    def raise_for_other_word(self, word, keywords):
        #word is not what the file should have here: it is either a cut keyword or content it does not know
        #it is cut if the file ends right after it, with no white space, and one of the keywords starts with it
        if self.at_end() and not self.text[-1:].isspace() and any(name.startswith(word.upper()) for name in keywords):
            raise FileIsCut('it ends inside a keyword')
        raise UnknownContent()

    def next_word(self, section):
        if self.at_end():
            raise FileIsCut('the header of '+section+' is not complete')
        if not self.is_binary:
            word=self.text[self.word_start[self.word]:self.word_end[self.word]]
            self.word+=1
            return word
        match=WORD.match(self.text, self.position)
        self.position=match.end()
        return match.group()

    def next_int(self, section):
        word=self.next_word(section)
        if not word.isdigit():
            raise UnknownContent()
        return int(word)

    def skip_values(self, count, value_type, section):
        if not self.is_binary:
            if value_type == b'string':
                raise UnknownContent()
            self.word+=count
            if self.word > len(self.word_start):
                raise FileIsCut(section+' has fewer values than it declares')
            return
        if value_type in BYTES_OF_VALUE_TYPE:
            size=count*BYTES_OF_VALUE_TYPE[value_type]
        elif self.position >= len(self.text):
            #no line break after the type, so its name may be cut
            raise FileIsCut('the header of '+section+' is not complete')
        else:
            raise UnknownContent()
        if size == 0:
            return
        #as in VTK's reader, the data starts after the next line break, whatever else is on this line.
        #the data itself may look like white space
        line_break=self.text.find(b'\n', self.position)
        if line_break < 0:
            raise FileIsCut(section+' has fewer values than it declares')
        self.position=line_break+1+size
        if self.position > len(self.text):
            raise FileIsCut(section+' has fewer values than it declares')
#%%
def walk_legacy_vtk_file(cursor, major_version):
    n_value=0 #the number of nodes or elements of the current POINT_DATA or CELL_DATA
    has_cells=False
    has_cell_types=False
    while not cursor.at_end():
        word=cursor.next_word('the file')
        keyword=word.upper()
        section=word.decode('ascii', 'replace')
        if keyword == b'DATASET':
            cursor.next_word(section)
        elif keyword == b'POINTS':
            count=cursor.next_int(section)
            cursor.skip_values(3*count, cursor.next_word(section).lower(), section)
        elif keyword in CELL_SECTION:
            has_cells=has_cells or keyword == b'CELLS'
            count=cursor.next_int(section)
            size=cursor.next_int(section)
            if major_version >= 5:
                #the first number is the size of OFFSETS, then CONNECTIVITY has the node indices
                for name, values in ((b'OFFSETS', count), (b'CONNECTIVITY', size)):
                    word=cursor.next_word(section)
                    if word.upper() != name:
                        cursor.raise_for_other_word(word, (name,))
                    cursor.skip_values(values, cursor.next_word(section).lower(), section)
            else:
                cursor.skip_values(size, b'int', section)
        elif keyword == b'CELL_TYPES':
            has_cell_types=True
            cursor.skip_values(cursor.next_int(section), b'int', section)
        elif keyword in (b'POINT_DATA', b'CELL_DATA'):
            n_value=cursor.next_int(section)
        elif keyword == b'FIELD':
            cursor.next_word(section)
            for k in range(cursor.next_int(section)):
                name=cursor.next_word(section)
                if name.upper() in (b'METADATA', b'NULL_ARRAY'):
                    raise UnknownContent()
                section=name.decode('ascii', 'replace')
                n_component=cursor.next_int(section)
                n_tuple=cursor.next_int(section)
                cursor.skip_values(n_component*n_tuple, cursor.next_word(section).lower(), section)
        elif keyword == b'SCALARS':
            cursor.next_word(section)
            value_type=cursor.next_word(section).lower()
            word=cursor.next_word(section)
            n_component=1
            if word.upper() != b'LOOKUP_TABLE':
                if not word.isdigit():
                    cursor.raise_for_other_word(word, (b'LOOKUP_TABLE',))
                n_component=int(word)
                cursor.next_word(section)
            cursor.next_word(section)
            cursor.skip_values(n_value*n_component, value_type, section)
        elif keyword in VALUES_OF_ATTRIBUTE:
            cursor.next_word(section)
            cursor.skip_values(n_value*VALUES_OF_ATTRIBUTE[keyword], cursor.next_word(section).lower(), section)
        elif keyword == b'TEXTURE_COORDINATES':
            cursor.next_word(section)
            n_component=cursor.next_int(section)
            cursor.skip_values(n_value*n_component, cursor.next_word(section).lower(), section)
        elif keyword == b'COLOR_SCALARS':
            cursor.next_word(section)
            cursor.skip_values(n_value*cursor.next_int(section), b'unsigned_char', section)
        elif keyword == b'LOOKUP_TABLE':
            cursor.next_word(section)
            cursor.skip_values(4*cursor.next_int(section), b'unsigned_char', section)
        else:
            cursor.raise_for_other_word(word, KEYWORD)
    if has_cells and not has_cell_types:
        raise FileIsCut('CELLS has no CELL_TYPES')
#%%
def find_cut_in_legacy_vtk_file(filename):
    """Return what is missing from a legacy VTK file that ends too early, or None."""
    with open(filename, 'rb') as file:
        text=file.read()
    line=text.split(b'\n', 3)
    version=re.match(rb'\s*#\s*vtk\s+DataFile\s+Version\s+(\d+)', line[0], re.I)
    if version is None or len(line) < 4 or line[2].strip().upper() not in (b'ASCII', b'BINARY'):
        return None
    position=len(line[0])+len(line[1])+len(line[2])+3
    cursor=LegacyVtkFileCursor(text, position, line[2].strip().upper() == b'BINARY')
    try:
        walk_legacy_vtk_file(cursor, int(version.group(1)))
    except FileIsCut as e:
        return str(e)
    except UnknownContent:
        pass
    if not cursor.is_binary and not text[-1:].isspace():
        print('warning: the file has no line break at its end, so its last value may be cut: '+filename)
    return None
#%%
