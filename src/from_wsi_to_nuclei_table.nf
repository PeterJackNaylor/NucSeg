
CWD = System.getProperty("user.dir")


params.data_type = "cam"

params.data = "/data2/pnaylor/datasets/pathology/Camelyon2016/*/*/*.tif"
params.tissue_segmentation_model = "${CWD}/outputs/tissue_segmentation_model"
params.nucleus_segmentation_model = "${CWD}/outputs/nuclei_segmentation_model"

DATA = file(params.data)
TISSUE_SEG_MODEL = file(params.tissue_segmentation_model)
NUCLEUS_SEG_MODEL = file(params.nucleus_segmentation_model)

process SegmentTissue {
    containerOptions '-B /data:/data'
    publishDir "./outputs/wsi_to_table/wsi_checks/${s_without_ext}", mode: 'copy', overwrite: 'true', pattern: "*.png"
    input:
        file sample from DATA
        file model from TISSUE_SEG_MODEL
    output:
        set file(sample), file("${s_without_ext}_mask.png") into WSI_MASK
        file("${s_without_ext}__img.png")
        file("${s_without_ext}__overlay.png")
        file("${s_without_ext}_prob.png")
    script:
        s_without_ext = "${sample}".split("\\.")[0]
        template 'segmentation/tissue.py'
}

process TilePatient {
    containerOptions '-B /data:/data --nv'
    publishDir "./outputs/wsi_to_table/seg_checks/${s_without_ext}", mode: 'copy', overwrite: 'true', pattern: "*.png"

    input:
        set file(sample), file(mask) from WSI_MASK
        file model from NUCLEUS_SEG_MODEL
    output:
        set val(sample), file("segmented_tiles.npz") into BATCH_SEG
        // add some pngs    
    script:
        s_without_ext = "${sample}".split("\\.")[0]
        template 'segmentation/tilling.py'
}