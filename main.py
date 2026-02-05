import applications.diffusion.setup as diff
import applications.smart_melt.setup as smart_mlt

if __name__ == "__main__":
    case = 1
    if case == 0 : diff.run_case()
    if case == 1 : smart_mlt.run_case()
