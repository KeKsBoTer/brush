#import helpers

@group(0) @binding(0) var<storage, read> uniforms: helpers::RenderUniforms;
@group(0) @binding(1) var<storage, read> compact_gid_from_isect: array<u32>;
@group(0) @binding(2) var<storage, read> tile_offsets: array<u32>;
@group(0) @binding(3) var<storage, read> projected: array<helpers::ProjectedSplat>;

#ifdef BWD_INFO
    @group(0) @binding(4) var<storage, read_write> out_img: array<vec4f>;
    @group(0) @binding(5) var<storage, read> global_from_compact_gid: array<u32>;
    @group(0) @binding(6) var<storage, read_write> visible: array<f32>;
#else
    @group(0) @binding(4) var<storage, read_write> out_img: array<u32>;
    @group(0) @binding(5) var<storage, read_write> out_img_gradient: array<array<vec4<f32>,3>>;
#endif

var<workgroup> range_uniform: vec2u;

var<workgroup> local_batch: array<helpers::ProjectedSplat, helpers::TILE_SIZE>;

#ifdef BWD_INFO
    var<workgroup> load_gid: array<u32, helpers::TILE_SIZE>;
#endif

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
    let pix_id = pix_loc.x + pix_loc.y * uniforms.img_size.x;
    let pixel_coord = vec2f(pix_loc) + 0.5f;
    let tile_loc = vec2u(pix_loc.x / helpers::TILE_WIDTH, pix_loc.y / helpers::TILE_WIDTH);

    let tile_id = tile_loc.x + tile_loc.y * uniforms.tile_bounds.x;
    let inside = pix_loc.x < uniforms.img_size.x && pix_loc.y < uniforms.img_size.y;

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between the bin counts.
    range_uniform = vec2u(
        tile_offsets[tile_id * 2],
        tile_offsets[tile_id * 2 + 1],
    );

    // Stupid hack as Chrome isn't convinced the range variable is uniform, which it better be.
    let range = workgroupUniformLoad(&range_uniform);

    // current visibility left to render
    var T = 1.0;
    var pix_out = vec3f(0.0);
    var pix_grad_out = array<vec3f, 4>(
        vec3f(0.0), vec3f(0.0), vec3f(0.0), vec3f(0.0)
    );
    var alpha_acc: vec3f= vec3f(0.0);
    var done = !inside;

    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    for (var batch_start = range.x; batch_start < range.y; batch_start += helpers::TILE_SIZE) {
        // process gaussians in the current batch for this pixel
        let remaining = min(helpers::TILE_SIZE, range.y - batch_start);

        let load_isect_id = batch_start + local_idx;
        let compact_gid = compact_gid_from_isect[load_isect_id];

        workgroupBarrier();
        if local_idx < remaining {
            local_batch[local_idx] = projected[compact_gid];
            #ifdef BWD_INFO
                load_gid[local_idx] = global_from_compact_gid[compact_gid];
            #endif
        }
        workgroupBarrier();

        for (var t = 0u; !done && t < remaining; t++) {
            let proj = local_batch[t];

            let xy = vec2f(proj.xy_x, proj.xy_y);
            let conic = vec3f(proj.conic_x, proj.conic_y, proj.conic_z);
            let color = vec4f(proj.color_r, proj.color_g, proj.color_b, proj.color_a);

            let delta = xy - pixel_coord;
            let sigma = 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) + conic.y * delta.x * delta.y;
            let alpha = min(0.999f, color.a * exp(-sigma));

            if sigma >= 0.0f && alpha >= 1.0f / 255.0f {
                let next_T = T * (1.0 - alpha);

                if next_T <= 1e-4f {
                    done = true;
                    break;
                }

                let dg_dx = (- conic.x * delta.x - conic.y * delta.y);
                let dg_dy = (- conic.y * delta.x - conic.z * delta.y);
                let dg_dxy = -conic.y;

                let dalpha_dx = dg_dx * alpha;
                let dalpha_dy = dg_dy * alpha;
                let dalpha_dxy = (dg_dx * dg_dy  + dg_dxy) * alpha;


                #ifdef BWD_INFO
                    // Count visible if contribution is at least somewhat significant.
                    visible[load_gid[t]] = 1.0;
                #endif

                let vis = alpha * T;
                let color_rgb = max(color.rgb, vec3f(0.0));
                pix_out += color_rgb * vis;

                pix_grad_out[0] += color_rgb * (T*dalpha_dxy - alpha_acc[2]*alpha - alpha_acc[1]*dalpha_dx - alpha_acc[0]*dalpha_dy);
				pix_grad_out[1] += color_rgb * (T*dalpha_dx  - alpha_acc[0]*alpha);
				pix_grad_out[2] += color_rgb * (T*dalpha_dy  - alpha_acc[1]*alpha);


                alpha_acc.z = alpha_acc.z*(1.0 - alpha) + T * dalpha_dxy - alpha_acc.x * dalpha_dy - alpha_acc.y * dalpha_dx;
                alpha_acc.x = alpha_acc.x*(1.0 - alpha) + T * dalpha_dx;
                alpha_acc.y = alpha_acc.y*(1.0 - alpha) + T * dalpha_dy;

                T = next_T;
            }
        }
    }

    if inside {
        // Compose with background. Nb that color is already pre-multiplied
        // by definition.
        let final_color = vec4f(pix_out + T * uniforms.background.rgb, 1.0 - T);
        
        // TODO consider background color in gradient

        #ifdef BWD_INFO
            out_img[pix_id] = final_color;
        #else
            let colors_u = vec4u(clamp(final_color * 255.0, vec4f(0.0), vec4f(255.0)));
            // let colors_u = vec4u(vec4f(clamp((pix_grad_out[0]+1.)*0.5 * 255.0, vec3f(0.0), vec3f(255.0)),255.0));
            let packed: u32 = colors_u.x | (colors_u.y << 8u) | (colors_u.z << 16u) | (colors_u.w << 24u);
            out_img[pix_id] = packed;
            out_img_gradient[pix_id][0] = vec4f(pix_grad_out[0],alpha_acc.x);
            out_img_gradient[pix_id][1] = vec4f(pix_grad_out[1],alpha_acc.y);
            out_img_gradient[pix_id][2] = vec4f(pix_grad_out[2],alpha_acc.z);
        #endif
    }
}
