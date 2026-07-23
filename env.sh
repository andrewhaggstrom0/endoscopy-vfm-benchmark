# source this in any new shell: source env.sh
export BIGDIR=/engrfs/project/class/a.haggstrom
export HF_HOME=$BIGDIR/hf_cache_endo
export FRAMES=$BIGDIR/endoscopy/raw/cholec80_frames
export DINO=$(ls -d $BIGDIR/endoscopy/cache/dinov2_vits14/*/cholec80 2>/dev/null)
export CLIP=$(ls -d $BIGDIR/endoscopy/cache/clip_vitb16/*/cholec80 2>/dev/null)
export ENDO=$(ls -d $BIGDIR/endoscopy/cache/endovit_vitb16/*/cholec80 2>/dev/null)
export BIOM=$(ls -d $BIGDIR/endoscopy/cache/biomedclip_vitb16/*/cholec80 2>/dev/null)
export ALL_CACHES="dinov2_vits14=$DINO clip_vitb16=$CLIP endovit_vitb16=$ENDO biomedclip_vitb16=$BIOM"
