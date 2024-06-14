class Tracker:
    def __init__(self, name):
        super().__init__()

        self.name = name

    def init(self, rgb_img, sonar_img, rgb_gt, sonar_gt):
        raise NotImplementedError

    def track(self, rgb_img, s_img):
        raise NotImplementedError


class SOTracker(Tracker):

    def __init__(self, name):
        super().__init__(name)
        self.rgb_tracker = None
        self.sonar_tracker = None

    def init(self, rgb_img, sonar_img, rgb_gt, sonar_gt):
        self.rgb_tracker.init(rgb_img, rgb_gt)
        self.sonar_tracker.init(sonar_img, sonar_gt)

    def track(self, rgb_img, s_img):

        rgb_box, rgb_score = self.rgb_tracker.track(rgb_img)
        if rgb_score < 0.5:
            rgb_box = [0, 0, 0, 0]

        s_box, sonar_score = self.sonar_tracker.track(s_img)
        if sonar_score < 0.5:
            s_box = [0, 0, 0, 0]

        return rgb_box, s_box
