"""Index the verified official validation tar without extracting 50,000 JPEG files."""
from pathlib import Path
import hashlib
import io
import json
import os
import tarfile
from common import ROOT, own_path, sha256, write_json

SPEC=json.loads((ROOT/'imagenet_sources.lock.json').read_text())


def verify_archive(path):
    spec=SPEC['archives'][path.name]
    md5=hashlib.md5();sha=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            md5.update(block);sha.update(block)
    if md5.hexdigest()!=spec['md5'] or ('bytes' in spec and path.stat().st_size!=spec['bytes']):
        raise RuntimeError('Official archive checksum/length differs: '+path.name)
    return {'name':path.name,'bytes':path.stat().st_size,'md5':md5.hexdigest(),'sha256':sha.hexdigest()}


def prepare_validation(archives, output):
    import numpy as np
    from scipy.io import loadmat
    archives=own_path(archives,exists=True);output=own_path(output)
    if (output/'metadata.json').exists():
        raise RuntimeError('Use a new index directory; existing verified metadata is not overwritten.')
    output.mkdir(parents=True,exist_ok=True)
    receipts=[verify_archive(archives/name) for name in ['ILSVRC2012_devkit_t12.tar.gz','ILSVRC2012_img_val.tar']]
    with tarfile.open(archives/'ILSVRC2012_devkit_t12.tar.gz','r:gz') as source:
        names={Path(m.name).name:m for m in source.getmembers() if m.isfile()}
        synsets=loadmat(io.BytesIO(source.extractfile(names['meta.mat']).read()),squeeze_me=True)['synsets']
        # Match Torchvision: leaf synsets only, then official numeric ID -> WNID.
        rows=[tuple(row) for row in synsets]
        mapping={int(row[0]):str(row[1]) for row in rows if int(row[4])==0}
        numeric=[int(x) for x in source.extractfile(names['ILSVRC2012_validation_ground_truth.txt']).read().split()]
    if len(mapping)!=1000 or len(numeric)!=50000:
        raise RuntimeError('Official devkit class or validation count differs.')
    class_file=ROOT/'data/functional/class_index.json'
    classes=json.loads(class_file.read_text());wnid_to_index={value[0]:int(key) for key,value in classes.items()}
    if set(wnid_to_index)!=set(mapping.values()) or set(wnid_to_index.values())!=set(range(1000)):
        raise RuntimeError('Canonical 1000-way WNID mapping differs from the official devkit.')
    labels=np.array([wnid_to_index[mapping[label]] for label in numeric],dtype=np.int64)
    if not np.array_equal(np.bincount(labels,minlength=1000),np.full(1000,50)):
        raise RuntimeError('Expected exactly 50 official validation images per class.')
    offsets=np.full(50000,-1,dtype=np.int64);sizes=np.zeros(50000,dtype=np.int64)
    with tarfile.open(archives/'ILSVRC2012_img_val.tar','r:') as source:
        for member in source:
            if not member.isfile():continue
            name=Path(member.name).name
            if not name.startswith('ILSVRC2012_val_') or not name.endswith('.JPEG'):
                raise RuntimeError('Unexpected official validation member: '+name)
            index=int(name[len('ILSVRC2012_val_'):-len('.JPEG')])-1
            if index<0 or index>=50000 or offsets[index]!=-1:
                raise RuntimeError('Duplicate/out-of-range ImageNet validation image ID.')
            offsets[index],sizes[index]=member.offset_data,member.size
    if (offsets<0).any() or (sizes<=0).any():raise RuntimeError('Missing official validation images.')
    np.savez(output/'val_index.npz',offsets=offsets,sizes=sizes,labels=labels)
    (output/'class_index.json').write_bytes(class_file.read_bytes())
    metadata={'scope':'imagenet_2012_official_validation_only','validation_images':50000,
        'training_images':0,'classes':1000,'images_per_class':50,
        'storage':'seekable_original_official_tar','archives_path':archives.relative_to(ROOT).as_posix(),
        'archive_receipts':receipts,'index_sha256':sha256(output/'val_index.npz'),
        'class_index_sha256':sha256(output/'class_index.json'),
        'label_mapping':'official devkit numeric ID -> leaf WNID -> canonical 1000-way index',
        'full_training_data_claimed':False}
    write_json(output/'metadata.json',metadata)
    return metadata


class ValidationTarDataset:
    def __init__(self,index_dir):
        import numpy as np
        from torchvision import transforms
        self.root=own_path(index_dir,exists=True)
        self.metadata=json.loads((self.root/'metadata.json').read_text())
        if self.metadata['scope']!='imagenet_2012_official_validation_only':raise RuntimeError('Wrong evaluation scope.')
        if sha256(self.root/'val_index.npz')!=self.metadata['index_sha256']:
            raise RuntimeError('Validation index digest differs.')
        if sha256(self.root/'class_index.json')!=self.metadata['class_index_sha256']:
            raise RuntimeError('Class mapping digest differs.')
        self.archive=own_path(ROOT/self.metadata['archives_path']/'ILSVRC2012_img_val.tar',exists=True)
        expected=next(x for x in self.metadata['archive_receipts'] if x['name']==self.archive.name)
        if self.archive.stat().st_size!=expected['bytes']:raise RuntimeError('Validation tar length differs.')
        arrays=np.load(self.root/'val_index.npz',allow_pickle=False)
        self.offsets=arrays['offsets'];self.sizes=arrays['sizes'];self.labels=arrays['labels']
        if len(self.labels)!=50000:raise RuntimeError('Complete 50,000-image validation is required.')
        self.transform=transforms.Compose([transforms.Resize(256,interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
        self._handle=None;self._pid=None

    def __len__(self):return len(self.labels)

    def __getstate__(self):
        result=dict(self.__dict__);result['_handle']=None;result['_pid']=None;return result

    def __getitem__(self,index):
        from PIL import Image
        if self._handle is None or self._pid!=os.getpid():
            self._handle=self.archive.open('rb');self._pid=os.getpid()
        self._handle.seek(int(self.offsets[index]));content=self._handle.read(int(self.sizes[index]))
        if len(content)!=int(self.sizes[index]):raise RuntimeError('Incomplete original JPEG read.')
        with Image.open(io.BytesIO(content)) as image:result=self.transform(image.convert('RGB'))
        return result,int(self.labels[index]),index
