use std::fs::File;
use std::io::Read;
use std::ops::{Add, Mul, Sub};
use std::path::Path;
use std::thread::sleep;
use std::time::Duration;
use ::image::*;
use piston_window::*;
use rand::Rng;
use rayon::prelude::*;

struct RGBA([f64; 4]);

impl RGBA {
    fn new(r: f64, g: f64, b: f64, a: f64) -> RGBA {
        RGBA([r, g, b, a])
    }
}

impl Into<Rgba<u8>> for RGBA {
    fn into(self) -> Rgba<u8> {
        Rgba([
            (self.0[0] * 255.0) as u8,
            (self.0[1] * 255.0) as u8,
            (self.0[2] * 255.0) as u8,
            (self.0[3] * 255.0) as u8,
        ])
    }
}

#[derive(Debug, Copy, Clone)]
struct Vec2([f64; 2]);
#[derive(Debug, Copy, Clone)]
struct Vec3([f64; 3]);
#[derive(Debug, Copy, Clone)]
struct Vec3H([f64; 4]);
#[derive(Debug, Copy, Clone)]
struct Mat3H([[f64; 4]; 4]);

impl Vec2 {
    fn length(&self) -> f64 {
        (self.0[0] * self.0[0] + self.0[1] * self.0[1]).sqrt()
    }
}

impl Sub<Vec2> for Vec2 {
    type Output = Vec2;
    fn sub(self, rhs: Vec2) -> Vec2 {
        Vec2([self.0[0] - rhs.0[0], self.0[1] - rhs.0[1]])
    }
}

impl Vec3 {
    fn to_homogeneous(&self) -> Vec3H {
        Vec3H([self.0[0], self.0[1], self.0[2], 1.0])
    }

    fn dot(&self, other: &Vec3) -> f64 {
        self.0[0] * other.0[0] + self.0[1] * other.0[1] + self.0[2] * other.0[2]
    }

    fn length(&self) -> f64 {
        (self.0[0] * self.0[0] + self.0[1] * self.0[1] + self.0[2] * self.0[2]).sqrt()
    }

    fn cross(&self, other: &Vec3) -> Vec3 {
        let a = &self.0;
        let b = &other.0;
        Vec3([
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ])
    }

    fn normalize(&self) -> Vec3 {
        let l = self.length();
        Vec3([self.0[0] / l, self.0[1] / l, self.0[2] / l])
    }
}

impl Sub<Vec3> for Vec3 {
    type Output = Vec3;
    fn sub(self, rhs: Vec3) -> Vec3 {
        Vec3([self.0[0] - rhs.0[0], self.0[1] - rhs.0[1], self.0[2] - rhs.0[2]])
    }
}

impl Add<Vec3> for Vec3 {
    type Output = Vec3;
    fn add(self, rhs: Vec3) -> Vec3 {
        Vec3([self.0[0] + rhs.0[0], self.0[1] + rhs.0[1], self.0[2] + rhs.0[2]])
    }
}

impl Mul<Vec3> for f64 {
    type Output = Vec3;

    fn mul(self, rhs: Vec3) -> Vec3 {
        Vec3([self * rhs.0[0], self * rhs.0[1], self * rhs.0[2]])
    }
}

impl Vec3H {
    fn homogeneous_divide(&self) -> Vec3 {
        Vec3([self.0[0] / self.0[3], self.0[1] / self.0[3], self.0[2] / self.0[3]])
    }
}

impl Mat3H {
    fn rot_y(theta: f64) -> Mat3H {
        let c = theta.cos();
        let s = theta.sin();
        Mat3H([
            [c, 0.0, s, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-s, 0.0, c, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
    }
}

impl Mul<Vec3H> for Mat3H {
    type Output = Vec3H;

    fn mul(self, other: Vec3H) -> Vec3H {
        Vec3H([
            self.0[0][0] * other.0[0] + self.0[0][1] * other.0[1] + self.0[0][2] * other.0[2] + self.0[0][3] * other.0[3],
            self.0[1][0] * other.0[0] + self.0[1][1] * other.0[1] + self.0[1][2] * other.0[2] + self.0[1][3] * other.0[3],
            self.0[2][0] * other.0[0] + self.0[2][1] * other.0[1] + self.0[2][2] * other.0[2] + self.0[2][3] * other.0[3],
            self.0[3][0] * other.0[0] + self.0[3][1] * other.0[1] + self.0[3][2] * other.0[2] + self.0[3][3] * other.0[3],
        ])
    }
}

impl Mul<Mat3H> for Mat3H {
    type Output = Mat3H;

    fn mul(self, rhs: Mat3H) -> Mat3H {
        let mut result = [[0.0; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    result[i][j] += self.0[i][k] * rhs.0[k][j];
                }
            }
        }
        Mat3H(result)
    }
}

const epsilon: f64 = 0.00000001;

fn ray_triangle_intersect(orig: Vec3, dir: Vec3, (v0, v1, v2): (Vec3, Vec3, Vec3)) -> Option<f64> {
    let edge0 = v1 - v0;
    let edge1 = v2 - v0;
    let n = edge0.cross(&edge1);
    let area = n.length();

    let ndotdir = n.dot(&dir);
    if ndotdir.abs() < epsilon {
        return None;
    }

    let d = -n.dot(&v0);

    let t = -(n.dot(&orig) + d) / ndotdir;
    if t < 0.0 {
        return None;
    }

    let p = orig + t * dir;
    let vp0 = p - v0;
    let c = edge0.cross(&vp0);
    if n.dot(&c) < 0.0 {
        return None;
    }

    let vp1 = p - v1;
    let edge1 = v2 - v1;
    let c = edge1.cross(&vp1);
    if n.dot(&c) < 0.0 {
        return None;
    }

    let vp2 = p - v2;
    let edge2 = v0 - v2;
    let c = edge2.cross(&vp2);
    if n.dot(&c) < 0.0 {
        return None;
    }

    Some(t)
}

fn intersects_triangle(origin: Vec3, dir: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Option<(f64, f64, f64)> {
    let a_to_b = c - a;
    let a_to_c = b - a;

    // Begin calculating determinant - also used to calculate u parameter
    // u_vec lies in view plane
    // length of a_to_c in view_plane = |u_vec| = |a_to_c|*sin(a_to_c, dir)
    let u_vec = dir.cross(&a_to_c);

    // If determinant is near zero, ray lies in plane of triangle
    // The determinant corresponds to the parallelepiped volume:
    // det = 0 => [dir, a_to_b, a_to_c] not linearly independant
    let det = a_to_b.dot(&u_vec);

    // Only testing positive bound, thus enabling backface culling
    // If backface culling is not desired write:
    // det < EPSILON && det > -EPSILON
    if det < epsilon {
        return None;
    }

    let inv_det = 1.0 / det;

    // Vector from point a to ray origin
    let a_to_origin = origin - a;

    // Calculate u parameter
    let u = a_to_origin.dot(&u_vec) * inv_det;

    // Test bounds: u < 0 || u > 1 => outside of triangle
    if !(0.0..=1.0).contains(&u) {
        return None;
    }

    // Prepare to test v parameter
    let v_vec = a_to_origin.cross(&a_to_b);

    // Calculate v parameter and test bound
    let v = dir.dot(&v_vec) * inv_det;
    // The intersection lies outside of the triangle
    if v < 0.0 || u + v > 1.0 {
        return None;
    }

    let dist = a_to_c.dot(&v_vec) * inv_det;

    if dist > epsilon {
        Some((dist, u, v))
    } else {
        None
    }
}

fn render(width: u32, height: u32, canvas: &mut ImageBuffer<Rgba<u8>, Vec<u8>>) {
    let cam_pos = Vec3([0.0, 0.0, 0.0]);

    let fov_y = 80.0 * std::f64::consts::PI / 180.0;
    let aspect_ratio = width as f64 / height as f64;
    let h = 2.0 * (fov_y / 2.0).tan();
    let f = (fov_y / 2.0).tan().recip();

    let z_near = 0.1;
    let z_far = 5.0;

    let mut projection = Mat3H([[0.0; 4]; 4]);
    projection.0[0][0] = f / aspect_ratio;
    projection.0[1][1] = f;
    projection.0[2][2] = (z_far + z_near) / (z_near - z_far);
    projection.0[3][2] = -1.0;
    projection.0[2][3] = (2.0 * z_far * z_near) / (z_near - z_far);

    let mut rotate180y = Mat3H([[0.0; 4]; 4]);
    rotate180y.0[0][0] = -1.0;
    rotate180y.0[1][1] = 1.0;
    rotate180y.0[2][2] = -1.0;
    rotate180y.0[3][3] = 1.0;

    let rotate180y = Mat3H::rot_y(150.0 * std::f64::consts::PI / 180.0);


    dbg!(projection);

    // // confusing, because radius gets projected into non-uniform radius. instead, let's focus on vertices
    // let ball_center = Vec3([0.0, 0.0, -50.0]).to_homogeneous();
    // let ball_radius = 13.5;
    // let on_ball_border = Vec3([0.0, ball_radius, -50.0]).to_homogeneous();
    //
    // // let outside = Vec3([0.0, 49.0, -50.0]).to_homogeneous();
    //
    // let p_ball_center = projection * ball_center;
    // dbg!(p_ball_center);
    // let p_ball_center = p_ball_center.homogeneous_divide();
    //
    // let p_on_ball_border = projection * on_ball_border;
    // let p_on_ball_border = p_on_ball_border.homogeneous_divide();
    //
    // let projected_radius = (p_ball_center - p_on_ball_border).length();
    // dbg!(projected_radius);
    //
    // dbg!(p_ball_center);
    //
    // // dbg!((projection * outside).homogeneous_divide());

    // let v1 = Vec3([-2.0f64.sqrt()/2.0, -2.0f64.sqrt()/2.0, -5.0]).to_homogeneous();
    // let v2 = Vec3([2.0f64.sqrt()/2.0, -2.0f64.sqrt()/2.0, -5.0]).to_homogeneous();
    // let v3 = Vec3([0.0, 1.0, -5.0]).to_homogeneous();

    // let v1 = Vec3([-20.0, 25.0, -60.0]).to_homogeneous();
    // let v2 = Vec3([20.0, 25.0, -60.0]).to_homogeneous();
    // let v3 = Vec3([0.0, 10.0, -90.0]).to_homogeneous();
    //
    // let v4 = Vec3([0.0, -1.0, -1.5]).to_homogeneous();
    //
    //
    let light_pos = Vec3([2.0, 3.0, -0.5]).to_homogeneous();
    let light_pos = Vec3([0.0, 0.0, -0.5]).to_homogeneous();
    //
    // let v1_p = (projection * v1).homogeneous_divide();
    // let v2_p = (projection * v2).homogeneous_divide();
    // let v3_p = (projection * v3).homogeneous_divide();
    // let v4_p = (projection * v4).homogeneous_divide();
    let light_pos_p = (projection * light_pos).homogeneous_divide();
    //
    // dbg!(v1_p);
    // dbg!(v2_p);
    // dbg!(v3_p);
    // dbg!(v4_p);

    let (mut verts, mut triangles) = load_obj("spot_triangulated.obj");
    // dbg!(&verts);
    // dbg!(triangles);

    // let idx1 = verts.len();
    // verts.push(Vec3([0.0, 0.1, -2.3]));
    // let idx2 = verts.len();
    // verts.push(Vec3([-0.1, 0.0, -2.3]));
    // let idx3 = verts.len();
    // verts.push(Vec3([0.1, 0.0, -2.3]));
    //
    // triangles.push(Triangle([idx1, idx2, idx3]));

    let mut verts_p = verts.clone().into_iter().map(|v| {
        let rotated = (rotate180y * v.to_homogeneous()).homogeneous_divide();

        (projection * (rotated - Vec3([0.0, 0.0, 1.5])).to_homogeneous()).homogeneous_divide()
    }).collect::<Vec<_>>();

    for v in &verts_p {
        dbg!(v);
    }


    println!("");
    for v in &verts_p {
        println!("v {} {} {}", v.0[0], v.0[1], v.0[2]);
    }
    println!("");

    dbg!(light_pos_p);


    let mut rng = rand::thread_rng();

    // let text = ::image::io::Reader::open("text.png").unwrap().decode().unwrap();

    // for triangle in &triangles {
    //
    // }

    for u in 0..width {
        println!("Processing pixel column ({u},...)");

        for v in 0..height {

            println!("Processing coord ({},{})", u, v);

            let u_f = 2.0 * (u) as f64 / width as f64 - 1.0;
            // let u_f = 2.0 * (u as f64 + rng.gen::<f64>()) / width as f64 - 1.0;
            let v_f = 2.0 * (height - v) as f64 / height as f64 - 1.0;
            // let v_f = 2.0 * ((height - v) as f64 + rng.gen::<f64>()) / height as f64 - 1.0;

            let screen_vec = Vec2([u_f, v_f]);
            // let ball_center = Vec2([p_ball_center.0[0], p_ball_center.0[1]]);

            let orig = Vec3([u_f, v_f, -1.0]);
            let dir = Vec3([0.0, 0.0, 1.0]);

            // for &Triangle([vi1, vi2, vi3]) in &triangles {
            //     let v1 = verts_p[vi1];
            //     let v2 = verts_p[vi2];
            //     let v3 = verts_p[vi3];
            //
            //
            // }

            let least = triangles.par_iter().enumerate().filter_map(|(i, &Triangle([vi1, vi2, vi3]))| {
                let v1 = verts_p[vi1];
                let v2 = verts_p[vi2];
                let v3 = verts_p[vi3];

                // match ray_triangle_intersect(orig, dir, (v1, v2, v3)) {
                match intersects_triangle(orig, dir, v1, v2, v3) {
                    Some(t) => Some((t, i)),
                    None => None,
                }
            // }).fold(((f64::INFINITY, 0.0, 0.0), 0), |(mt, mi), (t, i)| if t.0 < mt.0 { (t, i) } else { (mt, mi) });
            }).reduce(|| ((f64::INFINITY, 0.0, 0.0), 0), |(mt, mi), (t, i)| if t.0 < mt.0 { (t, i) } else { (mt, mi) });

            let least = if least.0.0 == f64::INFINITY { None } else { Some(least) };

            let [red, green, blue] = if let Some(((t, u, v), triangle_idx)) = least {
                dbg!(t);

                // now check illumination of hit point
                let triangle = triangles[triangle_idx];
                let v1 = verts[triangle.0[0]];
                let v2 = verts[triangle.0[1]];
                let v3 = verts[triangle.0[2]];

                let hit_point = orig + t * dir;
                let normal = 1.0 * (v3 - v1).cross(&(v2 - v1)).normalize();
                dbg!(normal);
                let camera_dir = ( dir).normalize();
                // let light_dir = (light_pos_p - hit_point).normalize();
                let light_dir = Vec3([1.0, -1.0, 1.0]).normalize();

                let diffuse = normal.dot(&light_dir);
                dbg!(diffuse);
                let diffuse = diffuse.max(0.0);

                let ambient = 0.1;

                let r = 2.0 * (light_dir.dot(&normal)) * normal - light_dir;
                let specular = r.dot(&camera_dir).max(0.0).powf(10.0);

                [1.0 * diffuse + ambient + specular, specular, specular]
                // [1.0 * diffuse + ambient, 0.0, 0.0]
                // 2.0 - t
            } else {
                [0.0; 3]
            };

            // let red = if let Some((t, triangle_idx)) = least {
            //
            //     // now check illumination of hit point
            //     let triangle = triangles[triangle_idx];
            //     let v1 = verts_p[triangle.0[0]];
            //     let v2 = verts_p[triangle.0[1]];
            //     let v3 = verts_p[triangle.0[2]];
            //
            //     let hit_point = orig + t * dir;
            //     let normal = (v1 - v2).cross(&(v1 - v3)).normalize();
            //     let camera_dir = (-1.0 * dir).normalize();
            //     let light_dir = (light_pos_p - hit_point).normalize();
            //
            //     let diffuse = -normal.dot(&light_dir);
            //     // dbg!(diffuse);
            //     let diffuse = diffuse.max(0.0);
            //
            //     let ambient = 0.1;
            //
            //     let r = 2.0 * (light_dir.dot(&normal)) * normal - light_dir;
            //     let specular = r.dot(&camera_dir).max(0.0).powf(10.0);
            //
            //     1.0 * diffuse + ambient + specular
            //     // 2.0 - t
            // } else {
            //     0.0
            // };

            // let blue = if let Some(t) = ray_triangle_intersect(orig, dir, (v1_p, v2_p, v4_p)) {
            //     // println!("{t:?}");
            //
            //     // now check illumination of hit point
            //
            //     let hit_point = orig + t * dir;
            //     let normal = (v1_p - v2_p).cross(&(v1_p - v4_p)).normalize();
            //     // let camera = (-1.0 * dir).normalize();
            //     let light_dir = (light_pos_p - hit_point).normalize();
            //
            //     let diffuse = normal.dot(&light_dir);
            //
            //     // dbg!(diffuse);
            //     let diffuse = diffuse.max(0.0);
            //
            //
            //     // 2.0 - t
            //     1.0 * diffuse
            // } else {
            //     0.0
            // };
            // let blue = 0.0;


            // let mut r = ((screen_vec - ball_center).length() <= projected_radius) as u8 as f64;

            let rgba = RGBA::new(red, green, blue, 1.0);
            canvas.put_pixel(u, v, rgba.into());
        }
    }
}

fn main() {
    let (width, height) = (80, 120);

    let opengl = OpenGL::V3_2;
    let mut window: PistonWindow =
        WindowSettings::new("ray", (width, height))
            .exit_on_esc(true)
            .graphics_api(opengl)
            .build()
            .unwrap();

    let mut canvas = ImageBuffer::new(width, height);

    render(width, height, &mut canvas);


    let mut texture_context = window.create_texture_context();

    let mut texture: G2dTexture = Texture::from_image(
        &mut texture_context,
        &canvas,
        &TextureSettings::new()
    ).unwrap();

    while let Some(e) = window.next() {
        // println!("{:?}", e);
        if let Some(_) = e.render_args() {
            // println!("{:?}", canvas.get_pixel(1,1));
            // canvas.put_pixel(1,1,RGBA::new(1.0, 0.0, 0.0, 1.0).into());
            // canvas.put_pixel(1,1,Rgba([255, 0, 0, 255]));
            //
            // for x in 0..width {
            //     for y in 0..height {
            //         canvas.put_pixel(x, y, Rgba([x as u8, y as u8, 0, 255]));
            //     }
            // }

            // render(width, height, &mut canvas);
            //
            texture.update(&mut texture_context, &canvas).unwrap();
            window.draw_2d(&e, |c, g, device| {
                // Update texture before rendering.
                texture_context.encoder.flush(device);
                clear([1.0; 4], g);
                image(&texture, c.transform, g);
            });
            //
            // sleep(Duration::from_secs(5));
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Triangle([usize; 3]);

fn load_obj(filename: impl AsRef<Path>) -> (Vec<Vec3>, Vec<Triangle>) {
    let mut vertices = Vec::new();
    let mut triangles = Vec::new();

    let mut file = File::open(filename).unwrap();
    let mut content = String::new();
    file.read_to_string(&mut content).unwrap();

    for line in content.lines() {
        let parts = line.split_whitespace().collect::<Vec<_>>();

        match parts[0] {
            "v" => {
                let v = parts[1..].iter().map(|x| x.parse::<f64>().unwrap()).collect::<Vec<_>>();
                vertices.push(Vec3([v[0], v[1], v[2]]));
            },
            "f" => {
                let f = parts[1..].iter().map(|x| x.split("/").next().unwrap().parse::<usize>().unwrap()).collect::<Vec<_>>();
                triangles.push(Triangle([f[0] - 1, f[1] - 1, f[2] - 1]));
            },
            _ => (),
        }
    }

    (vertices, triangles)
}
