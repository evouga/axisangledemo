#include <igl/opengl/glfw/Viewer.h>
#include <igl/read_triangle_mesh.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <imgui/imgui.h>
#include <iostream>
#include <sstream>
#include <Eigen/Geometry>
#include <igl/png/writePNG.h>

Eigen::Matrix3d crossMatrix(const Eigen::Vector3d &v)
{
    Eigen::Matrix3d result;
    result << 0, -v[2], v[1],
        v[2], 0, -v[0],
        -v[1], v[0], 0;
    return result;
}

Eigen::Vector3d naiveInterpolate(const Eigen::Vector3d &aa1, const Eigen::Vector3d &aa2, double t)
{
    Eigen::Vector3d axis = (1.0 - t) * aa1 + t * aa2;
    axis.normalize();
    double ang1 = aa1.norm();
    double ang2 = aa2.norm();
    double ang = (1.0 - t) * ang1 + t * ang2;
    return axis * ang;
}

Eigen::Matrix3d rodrigues(const Eigen::Vector3d &aa)
{
    double angle = aa.norm();
    Eigen::Matrix3d R;
    R.setIdentity();
    if (std::fabs(angle) > 1e-6)
    {
        Eigen::Vector3d axis = aa / aa.norm();
        Eigen::Matrix3d M = crossMatrix(axis);
        R += std::sin(angle) * M + (1.0 - std::cos(angle)) * M * M;
    }
    return R;
}

Eigen::Vector3d perpToAxis(const Eigen::Vector3d &v)
{
    int mincoord = 0;
    double minval = std::numeric_limits<double>::infinity();
    for(int i=0; i<3; i++)
    {
        if(fabs(v[i]) < minval)
        {
            mincoord = i;
            minval = fabs(v[i]);
        }
    }
    Eigen::Vector3d other(0,0,0);
    other[mincoord] = 1.0;
    Eigen::Vector3d result = v.cross(other);
    result.normalize();
    return result;
}

Eigen::Vector3d axisAngle(const Eigen::Matrix3d &rotationMatrix)
{
    Eigen::Matrix3d I;
    I.setIdentity();
    Eigen::Matrix3d RminusI = rotationMatrix - I;

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(RminusI, Eigen::ComputeFullV);
    //assert(fabs(svd.singularValues()[2]) < 1e-8);
    Eigen::Vector3d axis = svd.matrixV().col(2);
    Eigen::Vector3d testAxis = perpToAxis(axis);
    Eigen::Vector3d resultAxis = rotationMatrix*testAxis;
    double theta = atan2(testAxis.cross(resultAxis).dot(axis), testAxis.dot(resultAxis));
    return theta*axis;
}

Eigen::Vector3d correctInterpolate(const Eigen::Vector3d &aa1, const Eigen::Vector3d &aa2, double t)
{
    Eigen::Matrix3d R1 = rodrigues(aa1);
    Eigen::Matrix3d R2 = rodrigues(aa2);
    Eigen::Matrix3d diff = R2 * R1.transpose();
    Eigen::Vector3d aa = axisAngle(diff);
    aa *= t;
    Eigen::Matrix3d newR = rodrigues(aa) * R1;
    return axisAngle(newR);
}



Eigen::MatrixXd baseV;
Eigen::MatrixXi baseF;
Eigen::Vector3d axisangle1;
Eigen::Vector3d axisangle2;
int frame;
int nframes;
bool animate;
int method;
bool dumpframes;

void repaint(igl::opengl::glfw::Viewer &viewer)
{
    double t = double(frame) / double(nframes);
    Eigen::Vector3d aa;
    if (method == 0)
        aa = correctInterpolate(axisangle1, axisangle2, t);
    else
        aa = naiveInterpolate(axisangle1, axisangle2, t);
    Eigen::Matrix3d R = rodrigues(aa);
    Eigen::MatrixXd rotV = baseV * R.transpose();
    viewer.data().set_mesh(rotV, baseF);
    viewer.data().compute_normals();    
}

int main(int argc, char *argv[])
{
    
    if (!igl::read_triangle_mesh("bunny.obj", baseV, baseF))
    {
        if (!igl::read_triangle_mesh("../bunny.obj", baseV, baseF))
        {
            std::cerr << "Can't load bunny" << std::endl;
            return -1;
        }
    }
    axisangle1 << 2.0, 0, 0;
    axisangle2 << 0, 2.0, 0.0;
    frame = 0;
    nframes = 100;
    animate = false;
    dumpframes = false;
    // Plot the mesh
    igl::opengl::glfw::Viewer viewer;
    viewer.data().set_mesh(baseV, baseF);
    viewer.data().set_face_based(true);

    // Attach a menu plugin
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    viewer.plugins.push_back(&menu);

    menu.callback_draw_viewer_menu = [&]()
    {
        bool dirty = false;
        float aa1[3];
        for (int i = 0; i < 3; i++)
        {
            aa1[i] = (float)axisangle1[i];
        }
        if (ImGui::InputFloat3("AxisAngle #1", aa1))
        {
            for (int i = 0; i < 3; i++)
                axisangle1[i] = (double)aa1[i];
            dirty = true;
        }
        float aa2[3];
        for (int i = 0; i < 3; i++)
        {
            aa2[i] = (float)axisangle2[i];
        }
        if (ImGui::InputFloat3("AxisAngle #2", aa2))
        {
            for (int i = 0; i < 3; i++)
                axisangle2[i] = (double)aa2[i];
            dirty = true;
        }

        if (ImGui::Combo("Axis-Angle Interpolation Method", &method, "Correct\0Linear\0\0"))
        {
            dirty = true;
        }

        if (ImGui::Checkbox("Animate", &animate))
        {
            dirty = true;
        }

        if (ImGui::Button("Reset", ImVec2(-1, 0)))
        {
            frame = 0;
            dirty = true;
        }

        ImGui::Checkbox("Dump Frames", &dumpframes);
        
        if (dirty)
            repaint(viewer);
    };

    viewer.callback_post_draw = [&](igl::opengl::glfw::Viewer &viewer) -> bool
    {
        if (animate)
        {
            frame = (frame + 1) % nframes;
            repaint(viewer);

            if (dumpframes)
            {
                std::stringstream ss;
                ss << "frame_" << std::setw(6) << std::setfill('0') << frame << ".png";

                Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> R(1280,800);
                Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> G(1280,800);
                Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> B(1280,800);
                Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> A(1280,800);

                // Draw the scene in the buffers
                viewer.core().draw_buffer(
                    viewer.data(),false,R,G,B,A);

                // Save it to a PNG
                igl::png::writePNG(R,G,B,A,ss.str());
            }
        }

        
        return false;
    };

    viewer.core().is_animating = true;

    repaint(viewer);

    viewer.launch();
}
