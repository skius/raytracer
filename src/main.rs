use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::ops::{Add, Mul, Sub};
use std::path::Path;
use std::thread::sleep;
use std::time::{Duration, Instant};
use ::image::*;
use piston_window::*;
use rand::Rng;
use rayon::prelude::*;
use crate::types::Color;

struct RGBA([f64; 4]);

impl RGBA {
    fn new(r: f64, g: f64, b: f64, a: f64) -> RGBA {
        RGBA([r, g, b, a])
    }
}

impl Into<Rgba<u8>> for RGBA {
    fn into(self) -> Rgba<u8> {
        Rgba([
            (self.0[0] * 255.0).clamp(0.0, 255.0) as u8,
            (self.0[1] * 255.0).clamp(0.0, 255.0) as u8,
            (self.0[2] * 255.0).clamp(0.0, 255.0) as u8,
            (self.0[3] * 255.0).clamp(0.0, 255.0) as u8,
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

    fn rot_z(theta: f64) -> Mat3H {
        let c = theta.cos();
        let s = theta.sin();
        Mat3H([
            [c, s, 0.0, 0.0],
            [-s, c, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
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
    let a_to_b = b - a;
    let a_to_c = c - a;

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

fn interp_sky(t: f64) -> [f64; 3] {
    [0.5 * (1.0 - t) + 1.0 * t, 0.7 * (1.0 - t) + 1.0 * t, 1.0 * (1.0 - t) + 1.0 * t]
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum Kind {
    Phong,
    Mirror,
}

#[derive(Debug, Clone)]
struct Object {
    polygon_mesh: PolygonMesh,
    kind: Kind,
}

#[derive(Debug, Copy, Clone, Eq)]
struct Hit {
    distance: f64,
    u: f64,
    v: f64,
    triangle_idx: usize,
}

impl Hit {
    fn new(distance: f64, u: f64, v: f64, triangle_idx: usize) -> Hit {
        Hit {
            distance,
            u,
            v,
            triangle_idx,
        }
    }
}

impl PartialEq for Hit {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl PartialOrd for Hit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

impl Ord for Hit {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance.partial_cmp(&other.distance).unwrap()
    }
}

fn ray_hit_mesh(origin: Vec3, dir: Vec3, mesh: &PolygonMesh) -> Option<Hit> {
    let mut closest_hit = Hit::new(f64::INFINITY, 0.0, 0.0, 0);
    for (triangle_idx, triangle) in mesh.triangles.iter().enumerate() {
        if let Some((dist, u, v)) = intersects_triangle(origin, dir, triangle.0, triangle.1, triangle.2) {
            if dist < closest_hit.distance {
                closest_hit = Hit::new(dist, u, v, triangle_idx);
            }
        }
    }
    if closest_hit.distance < f64::INFINITY {
        Some(closest_hit)
    } else {
        None
    }
}

fn raytrace(origin: Vec3, dir: Vec3, objects: &[Object]) -> RGBA {
    // Check if it hits an object
    let closest_hit = Hit::new(f64::INFINITY, 0.0, 0.0, 0);
    let closest_obj_idx = usize::MAX;
    for (obj_idx, obj) in objects.iter().enumerate() {
        let hit = obj.polygon_mesh.hit(origin, dir);
        if let Some(hit) = hit {
            if hit.distance < closest_hit.distance {
                closest_hit = hit;
                closest_obj_idx = obj_idx;
            }
        }
    }

    []

}

#[derive(Debug, Clone)]
struct PolygonMesh {
    vertices: Vec<Vec3>,
    vertice_normals: Vec<Vec3>,
    uv_coords: Vec<(f64, f64)>,
    vertice_to_uv_idx: HashMap<usize, usize>,
    triangles: Vec<Triangle>,
    texture: DynamicImage,
}

impl PolygonMesh {
    fn rotate_y(&mut self, angle: f64) {
        let rot = Mat3H::rot_y(angle);

        for vertice in &mut self.vertices {
            *vertice = (rot * (*vertice).to_homogeneous()).homogeneous_divide();
        }
        for vertice_normal in &mut self.vertice_normals {
            *vertice_normal = (rot * (*vertice_normal).to_homogeneous()).homogeneous_divide();
        }
    }

    fn translate(&mut self, translation: Vec3) {
        for vertice in &mut self.vertices {
            *vertice = *vertice + translation;
        }
    }
}

fn render<const width: u32, const height: u32>(canvas: &mut ImageBuffer<Rgba<u8>, Vec<u8>>, time: f64, objects: &[Object]) {
    let fov_y = 60.0 * std::f64::consts::PI / 180.0;
    let aspect_ratio = width as f64 / height as f64;
    dbg!(time);
    let rotate180y = Mat3H::rot_y((150.0 + time) * std::f64::consts::PI / 180.0);
    // let rotate180y = Mat3H::rot_y((150.0) * std::f64::consts::PI / 180.0);

    let light_pos = Vec3([5.0, 5.0, 0.0]).to_homogeneous();

    let mut verts_p = polygon.vertices.clone().into_iter().map(|v| {
        let rotated = (rotate180y * v.to_homogeneous()).homogeneous_divide();
        // let rotated = (rotate180y * (Mat3H::rot_z(50.0 * (0.05*time).sin() * std::f64::consts::PI / 180.0) * v.to_homogeneous())).homogeneous_divide();

        rotated - Vec3([1.0, 0.0, 2.5])
    }).collect::<Vec<_>>();

    let mut verts_p_glossy = verts.clone().into_iter().map(|v| {
        let rotated = (Mat3H::rot_y((220.0) * std::f64::consts::PI / 180.0) * v.to_homogeneous()).homogeneous_divide();

        rotated - Vec3([-1.0, 0.0, 2.5])
    }).collect::<Vec<_>>();

    let verts_n_glossy = verts_n.clone().into_iter().map(|v| {
        let rotated = (Mat3H::rot_y((220.0) * std::f64::consts::PI / 180.0) * v.to_homogeneous()).homogeneous_divide();

        rotated
    }).collect::<Vec<_>>();

    let verts_n = verts_n.clone().into_iter().map(|v| {
        let rotated = (rotate180y * v.to_homogeneous()).homogeneous_divide();

        rotated
    }).collect::<Vec<_>>();

    let text_width = text.width() as usize;
    let text_height = text.height() as usize;


    for u in 0..width/2 {
        // println!("Processing pixel column ({u},...)");

        for v in 0..height {

            // println!("Processing coord ({},{})", u, v);


            let p_x = (2.0 * ((u as f64 + 0.5) / width as f64) - 1.0) * (fov_y / 2.0).tan() * aspect_ratio;
            let p_y = (1.0 - 2.0 * ((v as f64 + 0.5) / height as f64)) * (fov_y / 2.0).tan();

            let orig = Vec3([0.0, 0.0, 0.0]);
            let dir = Vec3([p_x, p_y, -1.0]).normalize();

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
                // dbg!(t);

                // now check illumination of hit point
                let triangle = triangles[triangle_idx];
                let v1 = verts_p[triangle.0[0]];
                let v2 = verts_p[triangle.0[1]];
                let v3 = verts_p[triangle.0[2]];

                let Triangle([vi1, vi2, vi3]) = triangle;

                let hit_point = orig + t * dir;
                // let normal = 1.0 * (v2 - v1).cross(&(v3 - v1)).normalize();

                let normal = (1.0 - u - v) * verts_n[vi1] + u * verts_n[vi2] + v * verts_n[vi3];

                // dbg!(verts_n[vi1]);
                // dbg!(verts_n[vi2]);
                // dbg!(verts_n[vi3]);
                // dbg!(u);
                // dbg!(v);
                //
                // dbg!(normal);
                let camera_dir = (-1.0 * dir).normalize();
                let light_dir = (light_pos.homogeneous_divide() - hit_point).normalize();
                // let light_dir = Vec3([1.0, -1.0, 1.0]).normalize();

                let diffuse = normal.dot(&light_dir);
                // dbg!(diffuse);
                let diffuse = diffuse.max(0.0);

                let ambient = 0.1;

                let r = 2.0 * (light_dir.dot(&normal)) * normal - light_dir;
                let specular = r.dot(&camera_dir).max(0.0).powf(10.0);

                let (u1, v1) = uvs[verts_to_uv_idx[&vi1]];
                let (u2, v2) = uvs[verts_to_uv_idx[&vi2]];
                let (u3, v3) = uvs[verts_to_uv_idx[&vi3]];

                let text_u = (1.0 - u - v) * u1 + u * u2 + v * u3;
                let text_v = (1.0 - u - v) * v1 + u * v2 + v * v3;

                let text_x = (text_u * text_width as f64).floor() as usize;
                let text_y = text_height - (text_v * text_height as f64).floor() as usize;
                // dbg!((text_x, text_y));

                let Rgba([r, g, b, a]) = text.get_pixel(text_x as u32, text_y as u32);

                let r = r as f64 / 255.0;
                let g = g as f64 / 255.0;
                let b = b as f64 / 255.0;

                [(diffuse + ambient + specular) * r, (diffuse + ambient + specular) * g, (diffuse + ambient + specular) * b]
                // [1.0 * diffuse + ambient, 0.0, 0.0]
                // 2.0 - t
            } else {
                let t = dir.0[1];
                interp_sky(t)
            };

            let rgba = RGBA::new(red, green, blue, 1.0);
            canvas.put_pixel(u, v, rgba.into());
        }
    }

    for u in width/2..width {
        // println!("Processing pixel column ({u},...)");

        for v in 0..height {

            // println!("Processing coord ({},{})", u, v);


            let p_x = (2.0 * ((u as f64 + 0.5) / width as f64) - 1.0) * (fov_y / 2.0).tan() * aspect_ratio;
            let p_y = (1.0 - 2.0 * ((v as f64 + 0.5) / height as f64)) * (fov_y / 2.0).tan();

            let orig = Vec3([0.0, 0.0, 0.0]);
            let dir = Vec3([p_x, p_y, -1.0]).normalize();

            // for &Triangle([vi1, vi2, vi3]) in &triangles {
            //     let v1 = verts_p[vi1];
            //     let v2 = verts_p[vi2];
            //     let v3 = verts_p[vi3];
            //
            //
            // }

            let least = triangles.par_iter().enumerate().filter_map(|(i, &Triangle([vi1, vi2, vi3]))| {
                let v1 = verts_p_glossy[vi1];
                let v2 = verts_p_glossy[vi2];
                let v3 = verts_p_glossy[vi3];

                // match ray_triangle_intersect(orig, dir, (v1, v2, v3)) {
                match intersects_triangle(orig, dir, v1, v2, v3) {
                    Some(t) => Some((t, i)),
                    None => None,
                }
                // }).fold(((f64::INFINITY, 0.0, 0.0), 0), |(mt, mi), (t, i)| if t.0 < mt.0 { (t, i) } else { (mt, mi) });
            }).reduce(|| ((f64::INFINITY, 0.0, 0.0), 0), |(mt, mi), (t, i)| if t.0 < mt.0 { (t, i) } else { (mt, mi) });

            let least = if least.0.0 == f64::INFINITY { None } else { Some(least) };

            let [red, green, blue] = if let Some(((t, u, v), triangle_idx)) = least {
                // dbg!(t);

                // now check illumination of hit point
                let triangle = triangles[triangle_idx];
                let Triangle([vi1, vi2, vi3]) = triangle;

                let hit_point = orig + t * dir;

                let normal = (1.0 - u - v) * verts_n_glossy[vi1] + u * verts_n_glossy[vi2] + v * verts_n_glossy[vi3];

                let camera_dir = (-1.0 * dir).normalize();

                let r = 2.0 * (camera_dir.dot(&normal)) * normal - camera_dir;

                let orig = hit_point;
                let dir = r.normalize();

                let light_dir = (light_pos.homogeneous_divide() - hit_point).normalize();
                let specular = r.dot(&light_dir).max(0.0).powf(20.0);

                let (u1, v1) = uvs[verts_to_uv_idx[&vi1]];
                let (u2, v2) = uvs[verts_to_uv_idx[&vi2]];
                let (u3, v3) = uvs[verts_to_uv_idx[&vi3]];

                let text_u = (1.0 - u - v) * u1 + u * u2 + v * u3;
                let text_v = (1.0 - u - v) * v1 + u * v2 + v * v3;

                let text_x = (text_u * text_width as f64).floor() as usize;
                let text_y = text_height - (text_v * text_height as f64).floor() as usize;
                // dbg!((text_x, text_y));

                let Rgba([r, g, b, a]) = text.get_pixel(text_x as u32, text_y as u32);
                let r = r as f64 / 255.0;
                let g = g as f64 / 255.0;
                let b = b as f64 / 255.0;

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

                let [rr, rg, rb] = if let Some(((t, u, v), triangle_idx)) = least {
                    // dbg!(t);

                    // now check illumination of hit point
                    let triangle = triangles[triangle_idx];
                    let v1 = verts_p[triangle.0[0]];
                    let v2 = verts_p[triangle.0[1]];
                    let v3 = verts_p[triangle.0[2]];

                    let Triangle([vi1, vi2, vi3]) = triangle;

                    let hit_point = orig + t * dir;
                    // let normal = 1.0 * (v2 - v1).cross(&(v3 - v1)).normalize();

                    let normal = (1.0 - u - v) * verts_n[vi1] + u * verts_n[vi2] + v * verts_n[vi3];

                    // dbg!(verts_n[vi1]);
                    // dbg!(verts_n[vi2]);
                    // dbg!(verts_n[vi3]);
                    // dbg!(u);
                    // dbg!(v);
                    //
                    // dbg!(normal);
                    let camera_dir = (-1.0 * dir).normalize();
                    let light_dir = (light_pos.homogeneous_divide() - hit_point).normalize();
                    // let light_dir = Vec3([1.0, -1.0, 1.0]).normalize();

                    let diffuse = normal.dot(&light_dir);
                    // dbg!(diffuse);
                    let diffuse = diffuse.max(0.0);

                    let ambient = 0.1;

                    let r = 2.0 * (light_dir.dot(&normal)) * normal - light_dir;
                    let specular = r.dot(&camera_dir).max(0.0).powf(10.0);

                    let (u1, v1) = uvs[verts_to_uv_idx[&vi1]];
                    let (u2, v2) = uvs[verts_to_uv_idx[&vi2]];
                    let (u3, v3) = uvs[verts_to_uv_idx[&vi3]];

                    let text_u = (1.0 - u - v) * u1 + u * u2 + v * u3;
                    let text_v = (1.0 - u - v) * v1 + u * v2 + v * v3;

                    let text_x = (text_u * text_width as f64).floor() as usize;
                    let text_y = text_height - (text_v * text_height as f64).floor() as usize;
                    // dbg!((text_x, text_y));

                    let Rgba([r, g, b, a]) = text.get_pixel(text_x as u32, text_y as u32);

                    let r = r as f64 / 255.0;
                    let g = g as f64 / 255.0;
                    let b = b as f64 / 255.0;

                    [(diffuse + ambient + specular) * r, (diffuse + ambient + specular) * g, (diffuse + ambient + specular) * b]
                    // let t = dir.0[1];
                    // interp_sky(t)
                } else {
                    let t = dir.0[1];
                    interp_sky(t)
                };

                [rr * r + specular, rg * g + specular, rb * b + specular]
            } else {
                let t = dir.0[1];
                interp_sky(t)
            };

            let rgba = RGBA::new(red, green, blue, 1.0);
            canvas.put_pixel(u, v, rgba.into());
        }
    }
}

fn main() {
    const width: u32 = 160;
    const height: u32 = 120;
    // let (width, height) = (5*160, 5*120);
    // let (width, height) = (20, 30);

    let mut start = Instant::now();

    let opengl = OpenGL::V3_2;
    let mut window: PistonWindow =
        WindowSettings::new("ray", (width, height))
            .exit_on_esc(true)
            .graphics_api(opengl)
            .build()
            .unwrap();

    let mut canvas = ImageBuffer::new(width, height);

    let spot_poly = load_obj("spot_triangulated.obj", "spot_texture.png");

    let mut texture_context = window.create_texture_context();

    let mut texture: G2dTexture = Texture::from_image(
        &mut texture_context,
        &canvas,
        &TextureSettings::new()
    ).unwrap();

    let mut t = 0.0;

    while let Some(e) = window.next() {
        // println!("{:?}", e);
        if let Some(re) = e.render_args() {
            // println!("{:?}", canvas.get_pixel(1,1));
            // canvas.put_pixel(1,1,RGBA::new(1.0, 0.0, 0.0, 1.0).into());
            // canvas.put_pixel(1,1,Rgba([255, 0, 0, 255]));
            //
            // for x in 0..width {
            //     for y in 0..height {
            //         canvas.put_pixel(x, y, Rgba([x as u8, y as u8, 0, 255]));
            //     }
            // }

            dbg!(Instant::now().duration_since(start));
            dbg!(re.ext_dt);
            t += re.ext_dt;

            let mut objects = Vec::new();
            let mut phong = spot_poly.clone();
            phong.rotate_y((150.0 + time) * std::f64::consts::PI / 180.0);
            phong.translate(Vec3([-1.0, 0.0, -2.5]));
            objects.push(Object {
                polygon_mesh: phong,
                kind: Kind::Phong,
            });

            let mut mirror = spot_poly.clone();
            mirror.rotate_y((220) * std::f64::consts::PI / 180.0);
            mirror.translate(Vec3([1.0, 0.0, -2.5]));
            objects.push(Object {
                polygon_mesh: mirror,
                kind: Kind::Mirror,
            });


            render::<width, height>(&mut canvas, t * 50000.0, &objects);


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

fn load_obj(filename: impl AsRef<Path>, texturename: impl AsRef<Path>) -> PolygonMesh {
    let mut vertices = Vec::new();
    let mut triangles = Vec::new();
    let mut uvs = Vec::new();
    let mut vertices_t = HashMap::new();

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
                let f = parts[1..].iter().map(|x| x.split("/").map(|e| e.parse::<usize>().unwrap()).collect::<Vec<_>>()).collect::<Vec<_>>();
                let vi1 = f[0][0] - 1;
                let vi2 = f[1][0] - 1;
                let vi3 = f[2][0] - 1;

                let uv1 = f[0][1] - 1;
                let uv2 = f[1][1] - 1;
                let uv3 = f[2][1] - 1;

                triangles.push(Triangle([vi1, vi2, vi3]));

                if vertices_t.contains_key(&vi1) {
                    println!("ALREADY HAS: {}, NEW: {}", vertices_t[&vi1], uv1);
                }

                vertices_t.insert(vi1, uv1);
                vertices_t.insert(vi2, uv2);
                vertices_t.insert(vi3, uv3);
            },
            "vt" => {
                let v = parts[1..].iter().map(|x| x.parse::<f64>().unwrap()).collect::<Vec<_>>();
                uvs.push((v[0], v[1]));
            },
            _ => (),
        }
    }

    let mut verts_n = vec![Vec::new(); vertices.len()];

    for &Triangle([idx1, idx2, idx3]) in &triangles {
        let v1 = vertices[idx1];
        let v2 = vertices[idx2];
        let v3 = vertices[idx3];

        let normal = (v2 - v1).cross(&(v3 - v1)).normalize();
        verts_n[idx1].push(normal);
        verts_n[idx2].push(normal);
        verts_n[idx3].push(normal);
    }

    let mut verts_normals = Vec::new();
    for v in verts_n {
        let mut normal = Vec3([0.0, 0.0, 0.0]);
        for n in v {
            normal = normal + n;
        }
        normal = normal.normalize();
        verts_normals.push(normal);
    }

    PolygonMesh {
        vertices,
        vertice_normals: verts_normals,
        uv_coords: uvs,
        triangles,
        vertice_to_uv_idx: vertices_t,
        texture: ::image::io::Reader::open(texturename).unwrap().decode().unwrap(),
    }
}
