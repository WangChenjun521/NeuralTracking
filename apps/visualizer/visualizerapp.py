import os
import vtk
import sys

from apps.visualizer import utilities
from apps.visualizer.mesh import Mesh


class VisualizerApp:

    def __init__(self, output_path, start_frame_ix=16):
        self.__alt_pressed = False
        # self.inverse_camera_matrices = trajectory_loading.load_inverse_matrices(output_path)
        self.start_frame_ix = start_frame_ix
        self.output_path = output_path
        self.offset_cam = (0.2562770766576, 0.13962609403401335, -0.2113334598208764)
        colors = vtk.vtkNamedColors()
        self.current_frame = start_frame_ix

        # renderer & render window initialization
        self.renderer = vtk.vtkRenderer()
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)

        frame_count = utilities.get_output_frame_count(output_path)
        self.frame_index_upper_bound = frame_count

        # mesh setup
        self.canonical_mesh = Mesh(self.renderer, self.render_window, colors.GetColor3d("Peacock"))
        self.warped_live_mesh = Mesh(self.renderer, self.render_window, colors.GetColor3d("Green"))
        self.shown_mesh_index = 1

        self.meshes = [self.canonical_mesh, self.warped_live_mesh]
        self.mesh_names = ["canonical_mesh", "warped_live_mesh"]

        self.renderer.SetBackground(colors.GetColor3d("Black"))
        self.render_window.SetSize(1400, 900)
        self.render_window.SetWindowName('Allocation')

        # Interactor setup
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetInteractorStyle(None)
        self.interactor.SetRenderWindow(self.render_window)
        self.interactor.Initialize()

        self.interactor.AddObserver("KeyPressEvent", self.keypress)
        self.interactor.AddObserver("KeyReleaseEvent", self.key_release)
        self.interactor.AddObserver("LeftButtonPressEvent", self.button_event)
        self.interactor.AddObserver("LeftButtonReleaseEvent", self.button_event)
        self.interactor.AddObserver("RightButtonPressEvent", self.button_event)
        self.interactor.AddObserver("RightButtonReleaseEvent", self.button_event)
        self.interactor.AddObserver("MiddleButtonPressEvent", self.button_event)
        self.interactor.AddObserver("MiddleButtonReleaseEvent", self.button_event)
        self.interactor.AddObserver("MiddleButtonReleaseEvent", self.button_event)
        self.interactor.AddObserver("MouseWheelForwardEvent", self.button_event)
        self.interactor.AddObserver("MouseWheelBackwardEvent", self.button_event)
        self.interactor.AddObserver("MouseMoveEvent", self.mouse_move)

        self.rotating = False
        self.panning = False
        self.zooming = False

        # axes actor
        self.axes = vtk.vtkAxesActor()
        self.axes_widget = widget = vtk.vtkOrientationMarkerWidget()
        rgba = colors.GetColor4ub("Carrot")
        widget.SetOutlineColor(rgba[0], rgba[1], rgba[2])
        widget.SetOrientationMarker(self.axes)
        widget.SetInteractor(self.interactor)
        widget.SetViewport(0.0, 0.2, 0.2, 0.4)
        widget.SetEnabled(1)
        widget.InteractiveOn()

        self.camera = camera = self.renderer.GetActiveCamera()

        # front=[0, 0, -1],
        # lookat=[0, 0, 1.5],
        # up=[0, -1.0, 0],
        # zoom=0.7

        camera.SetPosition(self.offset_cam[0], self.offset_cam[1], self.offset_cam[2])
        camera.SetPosition(0, 0, -1)
        camera.SetViewUp(0, 1.0, 0)
        camera.SetFocalPoint(0, 0, 1.5)
        camera.SetClippingRange(0.01, 10.0)

        self.render_window.Render()
        # 2nd monitor from left
        self.render_window.SetPosition(1281, 187)
        # 4th monitor
        # self.render_window.SetPosition(5121, 75)
        # fullscreen

        self.set_frame(self.current_frame)
        self.show_mesh_at_index(0)
        # uncomment to start out with meshes invisible
        # self.meshes[self.shown_mesh_index].toggle_visibility()
        self.render_window.Render()
        self._pixel_labels_visible = False

    def load_frame_meshes(self, i_frame):
        canonical_path = os.path.join(self.output_path, f"{i_frame:06d}_canonical_mesh.ply")
        warped_live_path = os.path.join(self.output_path, f"{i_frame:06d}_warped_mesh.ply")
        self.canonical_mesh.update(canonical_path)
        self.warped_live_mesh.update(warped_live_path)

    def launch(self):
        # Start the event loop.
        self.interactor.Start()

    def show_mesh_at_index(self, i_mesh_to_show):
        print("Mesh:", self.mesh_names[i_mesh_to_show])
        self.shown_mesh_index = i_mesh_to_show
        i_mesh = 0
        for mesh in self.meshes:
            if i_mesh_to_show == i_mesh:
                mesh.show()
            else:
                mesh.hide()
            i_mesh += 1
        self.render_window.Render()

    def set_frame(self, i_frame):
        print("Frame:", i_frame)

        self.load_frame_meshes(i_frame)
        # self.canonical_blocks.set_frame(i_frame)
        # self.live_blocks.set_frame(i_frame)
        # self.rays.set_frame(i_frame)
        self.current_frame = i_frame

        self.render_window.Render()

    def advance_view(self):
        if self.shown_mesh_index == len(self.meshes) - 1:
            if self.current_frame < self.frame_index_upper_bound - 1:
                self.set_frame(self.current_frame + 1)
                self.show_mesh_at_index(0)
        else:
            self.show_mesh_at_index(self.shown_mesh_index + 1)

    def retreat_view(self):
        if self.shown_mesh_index == 0:
            if self.current_frame > 0:
                self.set_frame(self.current_frame - 1)
                self.show_mesh_at_index(len(self.meshes) - 1)
        else:
            self.show_mesh_at_index(self.shown_mesh_index - 1)

    # Handle the mouse button events.
    def button_event(self, obj, event):
        if event == "LeftButtonPressEvent":
            self.rotating = True
        elif event == "LeftButtonReleaseEvent":
            self.rotating = False
        elif event == "RightButtonPressEvent":
            self.panning = True
        elif event == "RightButtonReleaseEvent":
            self.panning = False
        elif event == "MiddleButtonPressEvent":
            self.zooming = True
        elif event == "MiddleButtonReleaseEvent":
            self.zooming = False
        elif event == "MouseWheelForwardEvent":
            self.dolly_step(1)
        elif event == "MouseWheelBackwardEvent":
            self.dolly_step(-1)

    # General high-level logic
    def mouse_move(self, obj, event):
        last_x_y_pos = self.interactor.GetLastEventPosition()
        last_x = last_x_y_pos[0]
        last_y = last_x_y_pos[1]

        x_y_pos = self.interactor.GetEventPosition()
        x = x_y_pos[0]
        y = x_y_pos[1]

        center = self.render_window.GetSize()
        center_x = center[0] / 2.0
        center_y = center[1] / 2.0

        if self.rotating:
            self.rotate(x, y, last_x, last_y)
        elif self.panning:
            self.pan(x, y, last_x, last_y, center_x, center_y)
        elif self.zooming:
            self.dolly(x, y, last_x, last_y)

    def rotate(self, x, y, last_x, last_y):
        speed = 0.5
        self.camera.Azimuth(speed * (last_x - x))
        self.camera.Elevation(speed * (last_y - y))
        self.camera.SetViewUp(0, 0, 0)
        self.render_window.Render()

    def pan(self, x, y, last_x, last_y, center_x, center_y):
        renderer = self.renderer
        camera = self.camera
        f_point = camera.GetFocalPoint()
        f_point0 = f_point[0]
        f_point1 = f_point[1]
        f_point2 = f_point[2]

        p_point = camera.GetPosition()
        p_point0 = p_point[0]
        p_point1 = p_point[1]
        p_point2 = p_point[2]

        renderer.SetWorldPoint(f_point0, f_point1, f_point2, 1.0)
        renderer.WorldToDisplay()
        d_point = renderer.GetDisplayPoint()
        focal_depth = d_point[2]

        a_point0 = center_x + (x - last_x)
        a_point1 = center_y + (y - last_y)

        renderer.SetDisplayPoint(a_point0, a_point1, focal_depth)
        renderer.DisplayToWorld()
        r_point = renderer.GetWorldPoint()
        r_point0 = r_point[0]
        r_point1 = r_point[1]
        r_point2 = r_point[2]
        r_point3 = r_point[3]

        if r_point3 != 0.0:
            r_point0 = r_point0 / r_point3
            r_point1 = r_point1 / r_point3
            r_point2 = r_point2 / r_point3

        camera.SetFocalPoint((f_point0 - r_point0) / 2.0 + f_point0,
                             (f_point1 - r_point1) / 2.0 + f_point1,
                             (f_point2 - r_point2) / 2.0 + f_point2)
        camera.SetPosition((f_point0 - r_point0) / 2.0 + p_point0,
                           (f_point1 - r_point1) / 2.0 + p_point1,
                           (f_point2 - r_point2) / 2.0 + p_point2)
        self.render_window.Render()

    def dolly(self, x, y, last_x, last_y):
        dolly_factor = pow(1.02, (0.5 * (y - last_y)))
        camera = self.camera
        if camera.GetParallelProjection():
            parallel_scale = camera.GetParallelScale() * dolly_factor
            camera.SetParallelScale(parallel_scale)
        else:
            camera.Dolly(dolly_factor)
            self.renderer.ResetCameraClippingRange()

        self.render_window.Render()

    def dolly_step(self, step):
        dolly_factor = pow(1.02, (10.0 * step))
        camera = self.camera
        if camera.GetParallelProjection():
            parallel_scale = camera.GetParallelScale() * dolly_factor
            camera.SetParallelScale(parallel_scale)
        else:
            camera.Dolly(dolly_factor)
            self.renderer.ResetCameraClippingRange()

        self.render_window.Render()

    def keypress(self, obj, event):
        key = obj.GetKeySym()
        print("Key:", key)
        if key == "q" or key == "Escape":
            obj.InvokeEvent("DeleteAllObjects")
            sys.exit()
        elif key == "bracketright":
            if self.current_frame < self.frame_index_upper_bound - 1:
                self.set_frame(self.current_frame + 1)
        elif key == "bracketleft":
            if self.current_frame > 0:
                self.set_frame(self.current_frame - 1)
        # elif key == "c":
        #     if self.__alt_pressed:
        #         self.rays.canonical_source_point_cloud.toggle_visibility()
        #         self.render_window.Render()
        #     else:
        #         print(self.renderer.GetActiveCamera().GetPosition())
        elif key == "Right":
            self.advance_view()
        elif key == "Left":
            self.retreat_view()
        elif key == "p":
            if self.__alt_pressed:
                print(self.render_window.GetPosition())
            else:
                pass
        elif key == "m":
            self.meshes[self.shown_mesh_index].toggle_visibility()
            self.render_window.Render()
        # elif key == "l":
        #     if self.__alt_pressed:
        #         self.rays.live_source_point_cloud.toggle_visibility()
        #     else:
        #         self.active_blocks.toggle_labels()
        #     self.render_window.Render()
        # elif key == "b":
        #     if self.__alt_pressed:
        #         blocks_visible = self.active_blocks.blocks_visible()
        #         labels_visible = self.active_blocks.labels_visible()
        #         if blocks_visible:
        #             self.active_blocks.hide_blocks()
        #         if labels_visible:
        #             self.active_blocks.hide_labels()
        #         self.active_blocks = self.live_blocks if self.active_blocks == self.canonical_blocks else self.canonical_blocks
        #         if blocks_visible:
        #             self.active_blocks.show_blocks()
        #         if labels_visible:
        #             self.active_blocks.show_labels()
        #     else:
        #         self.active_blocks.toggle_visibility()
        #     self.render_window.Render()
        elif key == "Alt_L" or key == "Alt_R":
            self.__alt_pressed = True

    def key_release(self, obj, event):
        key = obj.GetKeySym()
        if key == "Alt_L" or key == "Alt_R":
            self.__alt_pressed = False
