import copy
import time
import open3d as o3d


class CustomDrawGeometryWithKeyCallbackViewer():
    def __init__(self, geometry_dict, alignment_dict, corresp_set):
        self.added_source_pcd = True
        self.added_source_obj = False
        self.added_target_pcd = False
        self.added_source_graph = False
        self.added_target_graph = False
        self.added_warped_pcd = False

        self.added_both = False

        self.added_corresp = False
        self.added_weighted_corresp = False

        self.aligned = False

        self.rotating = False
        self.stop_rotating = False
        self.axes_visible = False

        self.source_pcd = geometry_dict["source_pcd"]
        self.source_obj = geometry_dict["source_obj"]
        self.target_pcd = geometry_dict["target_pcd"]
        self.source_graph = geometry_dict["source_graph"]
        self.target_graph = geometry_dict["target_graph"]
        self.warped_pcd = geometry_dict["warped_pcd"]
        self.mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])

        # align source to target
        self.valid_source_points_cached = alignment_dict["valid_source_points"]
        self.valid_source_colors_cached = copy.deepcopy(self.source_obj.colors)
        self.line_segments_unit_cached = alignment_dict["line_segments_unit"]
        self.line_lengths_cached = alignment_dict["line_lengths"]

        self.good_matches_set = corresp_set["good_matches_set"]
        self.good_weighted_matches_set = corresp_set["good_weighted_matches_set"]
        self.bad_matches_set = corresp_set["bad_matches_set"]
        self.bad_weighted_matches_set = corresp_set["bad_weighted_matches_set"]

        # correspondences lines
        self.corresp_set = corresp_set

    def remove_both_pcd_and_object(self, vis, ref):
        if ref == "source":
            if self.added_source_pcd:
                vis.remove_geometry(self.source_pcd)
                self.added_source_pcd = False

            if self.added_source_obj:
                vis.remove_geometry(self.source_obj)
                self.added_source_obj = False

        elif ref == "target":
            if self.added_target_pcd:
                vis.remove_geometry(self.target_pcd)
                self.added_target_pcd = False

    def clear_for(self, vis, ref):
        if self.added_both:
            if ref == "target":
                vis.remove_geometry(self.source_obj)
                self.added_source_obj = False

            self.added_both = False

    def get_name_of_object_to_record(self):
        if self.added_both:
            return "both"

        if self.added_source_graph:
            if self.added_source_pcd:
                return "source_pcd_with_graph"
            if self.added_source_obj:
                return "source_obj_with_graph"
        else:
            if self.added_source_pcd:
                return "source_pcd"
            if self.added_source_obj:
                return "source_obj"

        if self.added_target_pcd:
            return "target_pcd"

    def custom_draw_geometry_with_key_callback(self):

        def toggle_source_graph(vis):
            if self.source_graph is None:
                print("You didn't pass me a source graph!")
                return False

            param = vis.get_view_control().convert_to_pinhole_camera_parameters()

            if self.added_source_graph:
                for g in self.source_graph:
                    vis.remove_geometry(g)
                self.added_source_graph = False
            else:
                for g in self.source_graph:
                    vis.add_geometry(g)
                self.added_source_graph = True
            ctr = vis.get_view_control()
            ctr.convert_from_pinhole_camera_parameters(param)
            return False

        def toggle_target_graph(vis):
            if self.target_graph is None:
                print("You didn't pass me a target graph!")
                return False

            param = vis.get_view_control().convert_to_pinhole_camera_parameters()

            if self.added_target_graph:
                for g in self.target_graph:
                    vis.remove_geometry(g)
                self.added_target_graph = False
            else:
                for g in self.target_graph:
                    vis.add_geometry(g)
                self.added_target_graph = True

            ctr = vis.get_view_control()
            ctr.convert_from_pinhole_camera_parameters(param)

            return False

        def toggle_obj(vis):
            print("::toggle_obj")

            if self.added_both:
                print("-- will not toggle obj. First, press either S or T, for source or target pcd")
                return False

            param = vis.get_view_control().convert_to_pinhole_camera_parameters()

            # Add source obj
            if self.added_source_pcd:
                vis.add_geometry(self.source_obj)
                vis.remove_geometry(self.source_pcd)
                self.added_source_obj = True
                self.added_source_pcd = False
            # Add source pcd
            elif self.added_source_obj:
                vis.add_geometry(self.source_pcd)
                vis.remove_geometry(self.source_obj)
                self.added_source_pcd = True
                self.added_source_obj = False

            ctr = vis.get_view_control()
            ctr.convert_from_pinhole_camera_parameters(param)

            return False

        def toggle_warped_pcd(vis):
            param = vis.get_view_control().convert_to_pinhole_camera_parameters()

            if self.added_warped_pcd:
                vis.remove_geometry(self.warped_pcd)
                self.added_warped_pcd = False
            else:
                vis.add_geometry(self.warped_pcd)
                self.added_warped_pcd = True
            ctr = vis.get_view_control()
            ctr.convert_from_pinhole_camera_parameters(param)

            return False

        def toggle_axes(vis):
            print("::toggle_axes")

            param = vis.get_view_control().convert_to_pinhole_camera_parameters()
            if self.axes_visible:
                self.axes_visible = False
                vis.remove_geometry(self.mesh_frame)
            else:
                self.axes_visible = True
                vis.add_geometry(self.mesh_frame)
            ctr = vis.get_view_control()
            ctr.convert_from_pinhole_camera_parameters(param)

            return False

        def view_source(vis):
            print("::view_source")

            self.clear_for(vis, "source")

            param = vis.get_view_control().convert_to_pinhole_camera_parameters()

            if not self.added_source_pcd:
                vis.add_geometry(self.source_pcd)
                self.added_source_pcd = True

            if self.added_source_obj:
                vis.remove_geometry(self.source_obj)
                self.added_source_obj = False

            self.remove_both_pcd_and_object(vis, "target")

            ctr = vis.get_view_control()
            ctr.convert_from_pinhole_camera_parameters(param)

            return False

        def view_target(vis):
            print("::view_target")

            self.clear_for(vis, "target")

            param = vis.get_view_control().convert_to_pinhole_camera_parameters()

            if not self.added_target_pcd:
                vis.add_geometry(self.target_pcd)
                self.added_target_pcd = True

            self.remove_both_pcd_and_object(vis, "source")

            ctr = vis.get_view_control()
            ctr.convert_from_pinhole_camera_parameters(param)

            return False

        def view_both(vis):
            print("::view_both")

            param = vis.get_view_control().convert_to_pinhole_camera_parameters()

            if self.added_source_pcd:
                vis.add_geometry(self.source_obj)
                vis.remove_geometry(self.source_pcd)
                self.added_source_obj = True
                self.added_source_pcd = False

            if not self.added_source_obj:
                vis.add_geometry(self.source_obj)
                self.added_source_obj = True

            if not self.added_target_pcd:
                vis.add_geometry(self.target_pcd)
                self.added_target_pcd = True

            self.added_both = True

            ctr = vis.get_view_control()
            ctr.convert_from_pinhole_camera_parameters(param)

        def rotate(vis):
            print("::rotate")

            self.rotating = True
            self.stop_rotating = False

            total = 2094
            speed = 5.0
            n = int(total / speed)
            for _ in range(n):
                ctr = vis.get_view_control()

                if self.stop_rotating:
                    return False

                ctr.rotate(speed, 0.0)
                vis.poll_events()
                vis.update_renderer()

            ctr.rotate(0.4, 0.0)
            vis.poll_events()
            vis.update_renderer()

            return False

        def rotate_slightly_left_and_right(vis):
            print("::rotate_slightly_left_and_right")

            self.rotating = True
            self.stop_rotating = False

            if self.added_both and not self.aligned:
                moves = ['lo', 'lo', 'pitch_f', 'pitch_b', 'ri', 'ro']
                totals = [(2094 / 4) / 2, (2094 / 4) / 2, 2094 / 4, 2094 / 4, 2094 / 4, 2094 / 4]
                abs_speed = 5.0
                abs_zoom = 0.15
                abs_pitch = 5.0
                iters_to_move = [range(int(t / abs_speed)) for t in totals]
                stop_at = [True, True, True, True, False, False]
            else:
                moves = ['ro', 'li']
                total = 2094 / 4
                abs_speed = 5.0
                abs_zoom = 0.03
                iters_to_move = [range(int(total / abs_speed)), range(int(total / abs_speed))]
                stop_at = [True, False]

            for move_idx, move in enumerate(moves):
                if move == 'l' or move == 'lo' or move == 'li':
                    h_speed = abs_speed
                elif move == 'r' or move == 'ro' or move == 'ri':
                    h_speed = -abs_speed

                if move == 'lo':
                    zoom_speed = abs_zoom
                elif move == 'ro':
                    zoom_speed = abs_zoom / 4.0
                elif move == 'li' or move == 'ri':
                    zoom_speed = -abs_zoom

                for _ in iters_to_move[move_idx]:
                    ctr = vis.get_view_control()

                    if self.stop_rotating:
                        return False

                    if move == "pitch_f":
                        ctr.rotate(0.0, abs_pitch)
                        ctr.scale(-zoom_speed / 2.0)
                    elif move == "pitch_b":
                        ctr.rotate(0.0, -abs_pitch)
                        ctr.scale(zoom_speed / 2.0)
                    else:
                        ctr.rotate(h_speed, 0.0)
                        if move == 'lo' or move == 'ro' or move == 'li' or move == 'ri':
                            ctr.scale(zoom_speed)

                    vis.poll_events()
                    vis.update_renderer()

                # Minor halt before changing direction
                if stop_at[move_idx]:
                    time.sleep(0.5)

                    if move_idx == 0:
                        toggle_correspondences(vis)
                        vis.poll_events()
                        vis.update_renderer()

                    if move_idx == 1:
                        toggle_weighted_correspondences(vis)
                        vis.poll_events()
                        vis.update_renderer()

                    time.sleep(0.5)

            ctr.rotate(0.4, 0.0)
            vis.poll_events()
            vis.update_renderer()

            return False

        def align(vis):
            if not self.added_both:
                return False

            moves = ['lo', 'li', 'ro']
            totals = [(2094 / 4) / 2, (2094 / 4), 2094 / 4 + (2094 / 4) / 2]
            abs_speed = 5.0
            abs_zoom = 0.15
            abs_pitch = 5.0
            iters_to_move = [range(int(t / abs_speed)) for t in totals]
            stop_at = [True, True, True]

            # Paint source object yellow, to differentiate target and source better
            # self.source_obj.paint_uniform_color([0, 0.8, 0.506])
            self.source_obj.paint_uniform_color([1, 0.706, 0])
            vis.update_geometry(self.source_obj)
            vis.poll_events()
            vis.update_renderer()

            for move_idx, move in enumerate(moves):
                if move == 'l' or move == 'lo' or move == 'li':
                    h_speed = abs_speed
                elif move == 'r' or move == 'ro' or move == 'ri':
                    h_speed = -abs_speed

                if move == 'lo':
                    zoom_speed = abs_zoom
                elif move == 'ro':
                    zoom_speed = abs_zoom / 50.0
                elif move == 'li' or move == 'ri':
                    zoom_speed = -abs_zoom / 5.0

                for _ in iters_to_move[move_idx]:
                    ctr = vis.get_view_control()

                    if self.stop_rotating:
                        return False

                    if move == "pitch_f":
                        ctr.rotate(0.0, abs_pitch)
                        ctr.scale(-zoom_speed / 2.0)
                    elif move == "pitch_b":
                        ctr.rotate(0.0, -abs_pitch)
                        ctr.scale(zoom_speed / 2.0)
                    else:
                        ctr.rotate(h_speed, 0.0)
                        if move == 'lo' or move == 'ro' or move == 'li' or move == 'ri':
                            ctr.scale(zoom_speed)

                    vis.poll_events()
                    vis.update_renderer()

                # Minor halt before changing direction
                if stop_at[move_idx]:
                    time.sleep(0.5)

                    if move_idx == 0:
                        n_iter = 125
                        for align_iter in range(n_iter + 1):
                            p = float(align_iter) / n_iter
                            self.source_obj.points = o3d.utility.Vector3dVector(
                                self.valid_source_points_cached + self.line_segments_unit_cached * self.line_lengths_cached * p)
                            vis.update_geometry(self.source_obj)
                            vis.poll_events()
                            vis.update_renderer()

                    time.sleep(0.5)

            self.aligned = True

            return False

        def toggle_correspondences(vis):
            if not self.added_both:
                return False

            param = vis.get_view_control().convert_to_pinhole_camera_parameters()

            if self.added_corresp:
                vis.remove_geometry(self.good_matches_set)
                vis.remove_geometry(self.bad_matches_set)
                self.added_corresp = False
            else:
                vis.add_geometry(self.good_matches_set)
                vis.add_geometry(self.bad_matches_set)
                self.added_corresp = True

            # also remove weighted corresp
            if self.added_weighted_corresp:
                vis.remove_geometry(self.good_weighted_matches_set)
                vis.remove_geometry(self.bad_weighted_matches_set)
                self.added_weighted_corresp = False

            ctr = vis.get_view_control()
            ctr.convert_from_pinhole_camera_parameters(param)

            return False

        def toggle_weighted_correspondences(vis):
            if not self.added_both:
                return False

            param = vis.get_view_control().convert_to_pinhole_camera_parameters()

            if self.added_weighted_corresp:
                vis.remove_geometry(self.good_weighted_matches_set)
                vis.remove_geometry(self.bad_weighted_matches_set)
                self.added_weighted_corresp = False
            else:
                vis.add_geometry(self.good_weighted_matches_set)
                vis.add_geometry(self.bad_weighted_matches_set)
                self.added_weighted_corresp = True

            # also remove (unweighted) corresp
            if self.added_corresp:
                vis.remove_geometry(self.good_matches_set)
                vis.remove_geometry(self.bad_matches_set)
                self.added_corresp = False

            ctr = vis.get_view_control()
            ctr.convert_from_pinhole_camera_parameters(param)

            return False

        def reload_source_object(vis):
            self.source_obj.points = o3d.utility.Vector3dVector(self.valid_source_points_cached)
            self.source_obj.colors = self.valid_source_colors_cached

            if self.added_both:
                vis.update_geometry(self.source_obj)
                vis.poll_events()
                vis.update_renderer()

        key_to_callback = {}
        key_to_callback[ord("G")] = toggle_source_graph
        key_to_callback[ord("D")] = toggle_target_graph
        key_to_callback[ord("S")] = view_source
        key_to_callback[ord("T")] = view_target
        key_to_callback[ord("O")] = toggle_obj
        key_to_callback[ord("B")] = view_both
        key_to_callback[ord("C")] = toggle_correspondences
        key_to_callback[ord("W")] = toggle_weighted_correspondences
        key_to_callback[ord(",")] = rotate
        key_to_callback[ord(";")] = rotate_slightly_left_and_right
        key_to_callback[ord("A")] = align
        key_to_callback[ord("Z")] = reload_source_object
        key_to_callback[ord("X")] = toggle_axes
        key_to_callback[ord("Y")] = toggle_warped_pcd

        o3d.visualization.draw_geometries_with_key_callbacks([self.source_pcd], key_to_callback)
