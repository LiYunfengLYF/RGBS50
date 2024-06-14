class Sequence:
    def __init__(self, name, rgb_imgs, sonar_imgs, rgb_gt, sonar_gt):
        super().__init__()
        self.name = name

        # imgs_list
        self.rgb_imgs = rgb_imgs
        self.sonar_imgs = sonar_imgs

        # gt_list
        self.rgb_gt = rgb_gt
        self.sonar_gt = sonar_gt

        # gt_dir


    def __getitem__(self, item):
        data = {
            'light': self.rgb_imgs[item],
            'sonar': self.sonar_imgs[item],
            'rgb_gt': self.rgb_gt[item],
            's_gt': self.sonar_gt[item],
        }
        return data

    def __len__(self):
        return len(self.rgb_imgs)
