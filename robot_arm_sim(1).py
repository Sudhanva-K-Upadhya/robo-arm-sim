import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QSlider, QLineEdit, QSpinBox, QTableWidget, QTableWidgetItem,
    QGroupBox, QGridLayout, QScrollArea, QFrame, QHeaderView, QSizePolicy,
    QPushButton
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QColor
import pyqtgraph as pg
import pyqtgraph.opengl as gl


# ─── DH Transform ────────────────────────────────────────────────────────────
def dh_transform(a, alpha, d, theta):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,      sa,     ca,    d],
        [0,       0,      0,    1]
    ])


def forward_kinematics(dh_params, joint_angles):
    """Returns list of T matrices (base to each joint), final is EE"""
    T = np.eye(4)
    transforms = [T]
    for i, (a, alpha, d, _) in enumerate(dh_params):
        theta = joint_angles[i]
        Ti = dh_transform(a, np.radians(alpha), d, np.radians(theta))
        T = T @ Ti
        transforms.append(T)
    return transforms


def inverse_kinematics(dh_params, target_pos, initial_angles=None,
                        max_iter=1000, tol=1e-3, lr=0.3, damping=0.01):
    """
    Numerical IK via damped least-squares (Levenberg-Marquardt) Jacobian.
    Solves for joint angles that place the EE at target_pos (x, y, z).
    Returns (angles_deg, success, final_error).
    """
    n = len(dh_params)
    angles = np.zeros(n) if initial_angles is None else np.array(initial_angles, dtype=float)
    target = np.array(target_pos, dtype=float)

    for _ in range(max_iter):
        transforms = forward_kinematics(dh_params, angles)
        ee = transforms[-1][:3, 3]
        error = target - ee
        err_norm = float(np.linalg.norm(error))
        
        if err_norm < tol:
            return angles.tolist(), True, err_norm

        # Build 3×n Jacobian via finite differences (angles in degrees)
        eps = 0.01  # degrees — much more numerically stable
        J = np.zeros((3, n))
        for j in range(n):
            a_perturbed = angles.copy()
            a_perturbed[j] += eps
            ee_perturbed = forward_kinematics(dh_params, a_perturbed)[-1][:3, 3]
            J[:, j] = (ee_perturbed - ee) / eps  # units: m/degree

        # Damped least-squares update — dq is already in degrees
        lam2 = damping ** 2
        dq = J.T @ np.linalg.solve(J @ J.T + lam2 * np.eye(3), error)
        angles += lr * dq  
        angles = np.clip(angles, -180, 180)

    final_err = float(np.linalg.norm(target - forward_kinematics(dh_params, angles)[-1][:3, 3]))
    return angles.tolist(), final_err < tol, final_err


# ─── 3D View Widget ───────────────────────────────────────────────────────────
class RobotView3D(gl.GLViewWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumWidth(600)
        self.setCameraPosition(distance=10, elevation=30, azimuth=-45)
        self.setBackgroundColor('#0a0e1a')

        # Grid
        grid = gl.GLGridItem()
        grid.setSize(10, 10)
        grid.setSpacing(1, 1)
        grid.setColor((40, 60, 80, 100))
        self.addItem(grid)

        self.world_axis_items = []
        self._add_axes()

        self.arm_items = []
        self.joint_items = []
        self.joint_axis_items = []
        self.ee_item = None

    def _add_axes(self):
        ax_len = 1
        axes = [
            ([0,0,0],[ax_len,0,0], (220,60,60,255)),
            ([0,0,0],[0,ax_len,0], (60,200,80,255)),
            ([0,0,0],[0,0,ax_len], (60,120,220,255)),
        ]
        for p1, p2, col in axes:
            pts = np.array([p1, p2])
            line = gl.GLLinePlotItem(pos=pts, color=col, width=2, antialias=True)
            self.addItem(line)
            self.world_axis_items.append(line)

    def _draw_joint_axes(self, T, scale=0.25):
        """Draw X/Y/Z axes at each joint frame with labels"""
        from pyqtgraph.opengl import GLTextItem
        origin = T[:3, 3]
        items = []
        axis_info = [
            (0, (220, 60, 60, 200),  "X"),
            (1, (60, 200, 80, 200),  "Y"),
            (2, (60, 120, 220, 200), "Z"),
        ]
        for i, col, label in axis_info:
            direction = T[:3, i] * scale
            tip = origin + direction
            pts = np.array([origin, tip])
            line = gl.GLLinePlotItem(pos=pts, color=col, width=2, antialias=True)
            self.addItem(line)
            items.append(line)
            try:
                txt = GLTextItem(pos=tip, text=label, color=(col[0], col[1], col[2], 220))
                self.addItem(txt)
                items.append(txt)
            except Exception:
                pass
        return items

    def update_robot(self, transforms):
        # Remove old elements
        for item in self.arm_items + self.joint_items + self.joint_axis_items:
            self.removeItem(item)
        if self.ee_item:
            self.removeItem(self.ee_item)
            
        self.arm_items.clear()
        self.joint_items.clear()
        self.joint_axis_items.clear()

        positions = [T[:3, 3] for T in transforms]

        # Draw links
        for i in range(len(positions) - 1):
            pts = np.array([positions[i], positions[i+1]])
            r = 1.0 - i / max(len(positions), 1)
            col = (int(40 + 180*r), int(180 - 80*r), int(220 - 100*r), 255)
            line = gl.GLLinePlotItem(pos=pts, color=col, width=5, antialias=True)
            self.addItem(line)
            self.arm_items.append(line)

        # Draw joints + their local axes
        for i, pos in enumerate(positions[:-1]):
            md = gl.MeshData.sphere(rows=8, cols=8, radius=0.12)
            mesh = gl.GLMeshItem(meshdata=md, smooth=True, color=(1.0, 0.7, 0.1, 1.0),
                                  shader='shaded', glOptions='opaque')
            mesh.translate(*pos)
            self.addItem(mesh)
            self.joint_items.append(mesh)
            self.joint_axis_items.extend(self._draw_joint_axes(transforms[i]))

        # EE axes & sphere
        self.joint_axis_items.extend(self._draw_joint_axes(transforms[-1], scale=0.35))
        
        md = gl.MeshData.sphere(rows=10, cols=10, radius=0.16)
        ee = gl.GLMeshItem(meshdata=md, smooth=True, color=(0.1, 1.0, 0.5, 1.0),
                            shader='shaded', glOptions='opaque')
        ee.translate(*positions[-1])
        self.addItem(ee)
        self.ee_item = ee


# ─── Styled Widgets ───────────────────────────────────────────────────────────
DARK = "#0a0e1a"
PANEL = "#111827"
BORDER = "#1e3a5f"
ACCENT = "#00d4ff"
ACCENT2 = "#00ff9d"
TEXT = "#c8d8e8"
MUTED = "#4a6080"
WARN = "#ffa040"

STYLE = f"""
QMainWindow, QWidget {{ background-color: {DARK}; color: {TEXT}; font-family: 'Consolas', monospace; font-size: 12px; }}
QGroupBox {{ border: 1px solid {BORDER}; border-radius: 6px; margin-top: 14px; padding: 8px; font-size: 11px; color: {ACCENT}; font-weight: bold; letter-spacing: 1px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; }}
QTableWidget {{ background-color: {PANEL}; border: 1px solid {BORDER}; gridline-color: #1a2a3a; color: {TEXT}; selection-background-color: #1e3a5f; }}
QHeaderView::section {{ background-color: #0d1929; color: {ACCENT}; border: 1px solid {BORDER}; padding: 4px; font-size: 11px; font-weight: bold; }}
QSlider::groove:horizontal {{ height: 4px; background: #1a2a3a; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {ACCENT}; border: 1px solid {ACCENT}; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QLineEdit {{ background-color: #0d1929; border: 1px solid {BORDER}; border-radius: 3px; color: {ACCENT2}; padding: 2px 6px; }}
QLineEdit:focus {{ border: 1px solid {ACCENT}; }}
QSpinBox {{ background-color: #0d1929; border: 1px solid {BORDER}; border-radius: 3px; color: {ACCENT2}; padding: 2px 6px; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{ background: {PANEL}; width: 6px; }}
QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 3px; }}
QPushButton {{ background-color: #0d1929; border: 1px solid {ACCENT}; border-radius: 4px; color: {ACCENT}; padding: 5px 16px; font-weight: bold; letter-spacing: 1px; }}
QPushButton:hover {{ background-color: #1e3a5f; color: #ffffff; }}
QPushButton:pressed {{ background-color: {ACCENT}; color: {DARK}; }}
"""


class MatrixDisplay(QWidget):
    def __init__(self):
        super().__init__()
        layout = QGridLayout(self)
        layout.setSpacing(2)
        self.cells = []
        for r in range(4):
            row = []
            for c in range(4):
                lbl = QLabel("0.000")
                lbl.setAlignment(Qt.AlignCenter)
                lbl.setFont(QFont("Consolas", 10))
                lbl.setMinimumWidth(75)
                layout.addWidget(lbl, r, c)
                row.append(lbl)
            self.cells.append(row)

    def set_matrix(self, T):
        for r in range(4):
            for c in range(4):
                val = T[r, c]
                color = ACCENT2 if abs(val) > 0.001 else MUTED
                self.cells[r][c].setText(f"{val:8.4f}")
                self.cells[r][c].setStyleSheet(f"background: #0d1929; border: 1px solid {BORDER}; color: {color}; padding: 3px 6px; border-radius: 2px;")


class JointControl(QWidget):
    angle_changed = pyqtSignal()

    def __init__(self, idx):
        super().__init__()
        self._updating = False
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        lbl = QLabel(f"J{idx+1}")
        lbl.setFixedWidth(20)
        lbl.setStyleSheet(f"color: {WARN}; font-weight: bold;")
        layout.addWidget(lbl)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(-180, 180)
        self.slider.setValue(0)
        layout.addWidget(self.slider)

        self.edit = QLineEdit("0.0")
        self.edit.setFixedWidth(60)
        self.edit.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.edit)

        lbl2 = QLabel("°")
        lbl2.setFixedWidth(12)
        layout.addWidget(lbl2)

        self.slider.valueChanged.connect(self._slider_changed)
        self.edit.returnPressed.connect(self._edit_changed)
        self.edit.editingFinished.connect(self._edit_changed)

    def _slider_changed(self, val):
        if self._updating: return
        self._updating = True
        self.edit.setText(f"{val:.1f}")
        self._updating = False
        self.angle_changed.emit()

    def _edit_changed(self):
        if self._updating: return
        try:
            val = float(self.edit.text())
            val = max(-180, min(180, val))
            self._updating = True
            self.slider.setValue(int(round(val)))
            self.edit.setText(f"{val:.1f}")
            self._updating = False
            self.angle_changed.emit()
        except ValueError:
            pass

    def get_angle(self):
        try:
            return float(self.edit.text())
        except ValueError:
            return 0.0

    def reset(self):
        self._updating = True
        self.slider.setValue(0)
        self.edit.setText("0.0")
        self._updating = False


# ─── Main Window ──────────────────────────────────────────────────────────────
class RobotArmSim(QMainWindow):
    DEFAULT_DH = [
        [0.0,  90.0, 1.0,  0.0],
        [1.5,   0.0, 0.0,  0.0],
        [1.0,   0.0, 0.0,  0.0],
        [0.0,  90.0, 0.5,  0.0],
        [0.0, -90.0, 0.0,  0.0],
        [0.0,   0.0, 0.3,  0.0],
    ]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Robot Arm Simulator — DH Parameters & FK")
        self.showFullScreen()
        self.setStyleSheet(STYLE)

        self.joint_controls = []
        self._build_ui()
        self._update_dof(3)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── LEFT PANEL ──
        left_widget = QWidget()
        left_widget.setFixedWidth(480)
        left_widget.setStyleSheet(f"background: {PANEL}; border-right: 1px solid {BORDER};")
        left = QVBoxLayout(left_widget)
        left.setContentsMargins(12, 12, 12, 12)
        left.setSpacing(10)

        # Title
        title_row = QHBoxLayout()
        title = QLabel("⚙  ROBOT ARM SIMULATOR")
        title.setFont(QFont("Consolas", 14, QFont.Bold))
        title.setStyleSheet(f"color: {ACCENT}; letter-spacing: 3px; padding: 6px 0;")
        title_row.addWidget(title)
        title_row.addStretch()
        quit_btn = QPushButton("[ Q ] EXIT")
        quit_btn.setStyleSheet(f"color: {WARN}; background: #1a0d00; border: 1px solid {WARN};")
        quit_btn.clicked.connect(self.close)
        title_row.addWidget(quit_btn)
        left.addLayout(title_row)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color: {BORDER};")
        left.addWidget(sep)

        # DOF selector
        dof_box = QGroupBox("DEGREES OF FREEDOM")
        dof_lay = QHBoxLayout(dof_box)
        self.dof_spin = QSpinBox()
        self.dof_spin.setRange(1, 6)
        self.dof_spin.setValue(3)
        self.dof_spin.setFixedWidth(70)
        dof_lay.addWidget(QLabel("DOF:"))
        dof_lay.addWidget(self.dof_spin)
        dof_lay.addStretch()
        left.addWidget(dof_box)
        self.dof_spin.valueChanged.connect(self._update_dof)

        # DH Table
        dh_box = QGroupBox("DH PARAMETER TABLE")
        dh_lay = QVBoxLayout(dh_box)
        self.dh_table = QTableWidget(3, 5)
        self.dh_table.setHorizontalHeaderLabels(["Joint", "a (m)", "α (°)", "d (m)", "θ offset (°)"])
        self.dh_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.dh_table.verticalHeader().setVisible(False)
        self.dh_table.setFixedHeight(170)
        dh_lay.addWidget(self.dh_table)
        left.addWidget(dh_box)
        self.dh_table.itemChanged.connect(self._on_dh_changed)

        # Joint controls
        joint_box = QGroupBox("JOINT ANGLES")
        joint_outer = QVBoxLayout(joint_box)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(220)
        self.joint_container = QWidget()
        self.joint_layout = QVBoxLayout(self.joint_container)
        self.joint_layout.setSpacing(8)
        scroll.setWidget(self.joint_container)
        joint_outer.addWidget(scroll)

        # --- NEW: Split buttons layout ---
        reset_btn_layout = QHBoxLayout()
        reset_btn_layout.setSpacing(10)
        
        reset_joints_btn = QPushButton("⟳ RESET JOINTS")
        reset_joints_btn.clicked.connect(self._reset_joints)
        
        reset_view_btn = QPushButton("⛶ RESET VIEW")
        reset_view_btn.clicked.connect(self._reset_view)
        
        reset_btn_layout.addWidget(reset_joints_btn)
        reset_btn_layout.addWidget(reset_view_btn)
        
        joint_outer.addLayout(reset_btn_layout)
        left.addWidget(joint_box)
        # ---------------------------------

        # FK Matrix
        fk_box = QGroupBox("FORWARD KINEMATICS — T_base→EE")
        fk_lay = QVBoxLayout(fk_box)
        self.matrix_display = MatrixDisplay()
        fk_lay.addWidget(QLabel("Homogeneous Transformation Matrix (4×4)"))
        fk_lay.addWidget(self.matrix_display)

        # EE Position
        ee_lay = QHBoxLayout()
        for label, attr in [("x:", "ee_x"), ("y:", "ee_y"), ("z:", "ee_z")]:
            l = QLabel(label); l.setStyleSheet(f"color:{MUTED};")
            v = QLabel("0.000 m"); v.setStyleSheet(f"color:{ACCENT2}; font-weight:bold;")
            setattr(self, attr, v)
            ee_lay.addWidget(l); ee_lay.addWidget(v); ee_lay.addSpacing(8)
        fk_lay.addLayout(ee_lay)
        left.addWidget(fk_box)

        # IK Panel
        ik_box = QGroupBox("INVERSE KINEMATICS — TARGET POSITION")
        ik_lay = QVBoxLayout(ik_box)

        ik_coord_lay = QHBoxLayout()
        self.ik_inputs = {}
        for axis in ("X", "Y", "Z"):
            lbl = QLabel(f"{axis}:")
            inp = QLineEdit("0.000")
            inp.setFixedWidth(80)
            inp.setAlignment(Qt.AlignCenter)
            ik_coord_lay.addWidget(lbl); ik_coord_lay.addWidget(inp); ik_coord_lay.addSpacing(6)
            self.ik_inputs[axis] = inp
        ik_lay.addLayout(ik_coord_lay)

        self.ik_solve_btn = QPushButton("⦿  SOLVE IK")
        self.ik_solve_btn.clicked.connect(self._solve_ik)
        ik_lay.addWidget(self.ik_solve_btn)

        self.ik_status = QLabel("—")
        self.ik_status.setAlignment(Qt.AlignCenter)
        ik_lay.addWidget(self.ik_status)

        left.addWidget(ik_box)
        left.addStretch()

        # ── RIGHT PANEL (3D) ──
        self.view3d = RobotView3D()
        self.view3d.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        root.addWidget(left_widget)
        root.addWidget(self.view3d, stretch=1)

    def _update_dof(self, dof):
        self.dof = dof
        for jc in self.joint_controls: jc.setParent(None)
        self.joint_controls.clear()

        for i in range(dof):
            jc = JointControl(i)
            jc.angle_changed.connect(self._on_joints_changed)
            self.joint_layout.addWidget(jc)
            self.joint_controls.append(jc)

        self.dh_table.blockSignals(True)
        self.dh_table.setRowCount(dof)
        for i in range(dof):
            a, alpha, d, theta = self.DEFAULT_DH[i]
            self.dh_table.setItem(i, 0, QTableWidgetItem(f"J{i+1}"))
            self.dh_table.item(i, 0).setFlags(Qt.ItemIsEnabled)
            self.dh_table.item(i, 0).setForeground(QColor(WARN))
            self.dh_table.setItem(i, 1, QTableWidgetItem(str(a)))
            self.dh_table.setItem(i, 2, QTableWidgetItem(str(alpha)))
            self.dh_table.setItem(i, 3, QTableWidgetItem(str(d)))
            self.dh_table.setItem(i, 4, QTableWidgetItem(str(theta)))
        self.dh_table.blockSignals(False)

        self._recompute()

    def _get_dh_params(self):
        params = []
        for i in range(self.dof):
            try:
                a = float(self.dh_table.item(i, 1).text())
                alpha = float(self.dh_table.item(i, 2).text())
                d = float(self.dh_table.item(i, 3).text())
                theta_off = float(self.dh_table.item(i, 4).text())
                params.append([a, alpha, d, theta_off])
            except (ValueError, AttributeError):
                params.append([0, 0, 0, 0])
        return params

    def _on_dh_changed(self): self._recompute()
    def _on_joints_changed(self): self._recompute()

    def _reset_joints(self):
        for jc in self.joint_controls: jc.reset()
        self._recompute()
        
    def _reset_view(self):
        # Resets the 3D camera to its default starting position
        self.view3d.setCameraPosition(distance=10, elevation=30, azimuth=-45)

    def _recompute(self):
        dh = self._get_dh_params()
        angles = [jc.get_angle() for jc in self.joint_controls]

        transforms = forward_kinematics(dh, angles)
        T_ee = transforms[-1]

        self.matrix_display.set_matrix(T_ee)
        self.ee_x.setText(f"{T_ee[0,3]:.4f} m")
        self.ee_y.setText(f"{T_ee[1,3]:.4f} m")
        self.ee_z.setText(f"{T_ee[2,3]:.4f} m")

        self.view3d.update_robot(transforms)

    def _solve_ik(self):
        try:
            target = [float(self.ik_inputs[ax].text()) for ax in ("X", "Y", "Z")]
        except ValueError:
            self.ik_status.setText("✗  invalid target coordinates")
            self.ik_status.setStyleSheet(f"color:{WARN}; font-size:10px;")
            return

        dh = self._get_dh_params()
        initial = [jc.get_angle() for jc in self.joint_controls]

        best_angles, best_success, best_err = inverse_kinematics(dh, target, initial_angles=initial)

        if not best_success:
            for _ in range(10):
                rand_init = np.random.uniform(-180, 180, len(dh)).tolist()
                angles, success, err = inverse_kinematics(dh, target, initial_angles=rand_init)
                if err < best_err:
                    best_angles, best_success, best_err = angles, success, err
                if best_success: break

        if best_success:
            self.ik_status.setText(f"✓  converged  |  residual: {best_err:.4f} m")
            self.ik_status.setStyleSheet(f"color:{ACCENT2}; font-size:10px;")
        else:
            self.ik_status.setText(f"✗  did not converge  |  residual: {best_err:.4f} m")
            self.ik_status.setStyleSheet(f"color:{WARN}; font-size:10px;")

        for jc, ang in zip(self.joint_controls, best_angles):
            jc._updating = True
            jc.slider.setValue(int(round(ang)))
            jc.edit.setText(f"{ang:.1f}")
            jc._updating = False

        self._recompute()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Q, Qt.Key_Escape):
            self.close()
        super().keyPressEvent(event)


# ─── Entry ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pg.setConfigOptions(antialias=True)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = RobotArmSim()
    win.show()
    sys.exit(app.exec_())