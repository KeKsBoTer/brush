#import helpers

@group(0) @binding(0) var<storage, read> uniforms: helpers::RenderUniforms;
@group(0) @binding(1) var<storage, read> rgba_img: array<u32>;
@group(0) @binding(2) var<storage, read> rgba_img_gradient: array<array<vec4<f32>,3>>;
@group(0) @binding(3) var<storage, read_write> out_img: array<u32>;


fn spline_interp(z:mat2x2<f32>, dx:mat2x2<f32>, dy: mat2x2<f32>, dxy: mat2x2<f32>, p: vec2<f32>) -> f32
{
    let f = mat4x4<f32>(
        z[0][0], z[0][1], dy[0][0], dy[0][1],
        z[1][0], z[1][1], dy[1][0], dy[1][1],
        dx[0][0], dx[0][1], dxy[0][0], dxy[0][1],
        dx[1][0], dx[1][1], dxy[1][0], dxy[1][1]
    );
    let m = mat4x4<f32>(
        1., 0., 0., 0.,
        0., 0., 1., 0.,
        -3., 3., -2., -1.,
        2., -2., 1., 1.
    );
    let a = transpose(m) * f * (m);

    let tx = vec4<f32>(1., p.x, p.x * p.x, p.x * p.x * p.x);
    let ty = vec4<f32>(1., p.y, p.y * p.y, p.y * p.y * p.y);
    return dot(tx, a * ty);
}


// kernel function for rasterizing each tile
// each thread treats a single pixel
// each thread group uses the same gaussian data in a tile
@compute
@workgroup_size(helpers::TILE_SIZE, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3u,
    @builtin(local_invocation_index) local_idx: u32,
) {
    let pix_loc = helpers::map_1d_to_2d(global_id.x, uniforms.tile_bounds.x);
    let pix_id = pix_loc.x + pix_loc.y * uniforms.target_size.x;
    let pixel_coord = vec2f(pix_loc) + 0.5f;

    let inside = pix_loc.x < uniforms.target_size.x && pix_loc.y < uniforms.target_size.y;

    if !inside {
        return;
    }
    if rgba_img_gradient[pix_id][0].x > 10000.{
        return;
    }

    let pixel_coord_source_f = pixel_coord*vec2f(uniforms.img_size)/vec2f(uniforms.target_size);

    let p_frac = fract(pixel_coord_source_f);
    var z:  array<mat2x2<f32>,4>;
    var dx: array<mat2x2<f32>,4>;
    var dy: array<mat2x2<f32>,4>;
    var dxy:array<mat2x2<f32>,4>;

    for (var i = 0; i < 2; i++) {
        for (var j = 0; j < 2; j++) {
            let pixel_coord_source = vec2u(clamp(vec2i(pixel_coord_source_f) + vec2(i,j), vec2i(0), vec2i(uniforms.img_size) - vec2i(1)));

            let pix_id_source = pixel_coord_source.x + pixel_coord_source.y * uniforms.img_size.x;

            let color = rgba_img[pix_id_source];
            let z_v = unpack4x8unorm(color);
            var dx_v = rgba_img_gradient[pix_id_source][0];
            var dy_v = rgba_img_gradient[pix_id_source][1];
            var dxy_v = rgba_img_gradient[pix_id_source][2];

            for (var c = 0u; c < 4u; c++) {
                z[c][i][j] = z_v[c];
                dx[c][i][j] = dx_v[c];
                dy[c][i][j] = dy_v[c];
                dxy[c][i][j] = dxy_v[c];
            }
        }
    }

    let color_interp = vec3<f32>(
        spline_interp(z[0], dx[0], dy[0], dxy[0], vec2<f32>(p_frac.y, p_frac.x)),
        spline_interp(z[1], dx[1], dy[1], dxy[1], vec2<f32>(p_frac.y, p_frac.x)),
        spline_interp(z[2], dx[2], dy[2], dxy[2], vec2<f32>(p_frac.y, p_frac.x)),
    ); 

    out_img[pix_id] = pack4x8unorm(vec4<f32>(color_interp, 1.0));
}
