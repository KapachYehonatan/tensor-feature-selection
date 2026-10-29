# Extract public sparse RDS slots with base R; Matrix/ExperimentHub are not required.
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2)
x <- tryCatch(readRDS(args[[1]]), error = function(error) {
    environment <- new.env()
    objects <- load(args[[1]], envir = environment)
    stopifnot(length(objects) == 1)
    environment[[objects[[1]]]]
})
out <- args[[2]]
dir.create(out, recursive = TRUE, showWarnings = FALSE)
kind <- class(x)[[1]]
stopifnot(kind %in% c("dgCMatrix", "dgTMatrix"))
dim <- attr(x, "Dim")
names <- attr(x, "Dimnames")
stopifnot(length(dim) == 2, length(names[[1]]) == dim[[1]], length(names[[2]]) == dim[[2]])
writeLines(kind, file.path(out, "class.txt"))
writeLines(as.character(dim), file.path(out, "shape.txt"))
writeLines(names[[1]], file.path(out, "features.txt"))
writeLines(names[[2]], file.path(out, "samples.txt"))
for (slot in c("i", if (kind == "dgCMatrix") "p" else "j", "x")) {
    con <- file(file.path(out, paste0(slot, ".bin")), "wb")
    writeBin(attr(x, slot), con, size = if (slot == "x") 8 else 4, endian = "little")
    close(con)
}
writeLines("complete", file.path(out, "complete.txt"))
