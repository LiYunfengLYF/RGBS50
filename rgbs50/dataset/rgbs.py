import os
from .data import Sequence
from ..utils.read import seqread, txtread


class RGBS50(object):
    _sequence_name = ['connected_polyhedron9', 'fake_person2', 'uuv7', 'iron_ball2', 'connected_polyhedron4',
                      'octahedron4', 'ball_and_polyhedron2', 'ball_and_polyhedron3', 'connected_polyhedron5',
                      'octahedron3', 'connected_polyhedron3', 'fake_person6', 'uuv9', 'uuv12', 'fake_person5', 'uuv5',
                      'uuv4', 'frustum5', 'octahedron6', 'connected_polyhedron7', 'octahedron1', 'iron_ball1',
                      'frustum6', 'frustum4', 'octahedron2', 'fake_person1', 'frustum1', 'uuv6', 'iron_ball3',
                      'frustum7', 'uuv1', 'connected_polyhedron1', 'connected_polyhedron6', 'uuv2', 'uuv3', 'uuv8',
                      'octahedron7', 'ball_and_polyhedron1', 'fake_person4', 'octahedron8', 'fake_person3', 'frustum2',
                      'octahedron5', 'fake_person7', 'uuv10', 'frustum3', 'connected_polyhedron8', 'frustum8', 'uuv11',
                      'connected_polyhedron2']

    def __init__(self, root_dir):
        super().__init__()

        self.name = 'RGBS50'
        self.root_dir = root_dir
        self.sequence_list = self.construct_sequence_list()

    def __getitem__(self, item):
        return self.sequence_list[item]

    def __len__(self):
        return len(self.sequence_list)

    def construct_sequence_list(self):
        sequence_list = []
        for name in self._sequence_name:
            sensor_rgb = os.path.join(self.root_dir, name, 'light', )
            sensor_s = os.path.join(self.root_dir, name, 'sonar')

            rgb_imgs = seqread(os.path.join(sensor_rgb, 'img'))
            s_imgs = seqread(os.path.join(sensor_s, 'img'))

            rgb_gt = txtread(os.path.join(sensor_rgb, 'groundtruth.txt'))
            s_gt = txtread(os.path.join(sensor_s, 'groundtruth.txt'))
            sequence_list.append(Sequence(name, rgb_imgs, s_imgs, rgb_gt, s_gt))
        return sequence_list

