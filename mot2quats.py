# For OpenSim support
import math, os, sys, getopt, numpy
import opensim as osim
import csv

def mot2quats(motionPath, outputPath, modelPath, optionsDict):

    print(f"Input OpenSim motion path set to: {motionPath}")

    model = osim.Model(modelPath)
    model.initSystem()

    bodyNames = []
    for body in model.getBodyList():
        bodyNames.append(body.getName())

    bodySet = model.getBodySet()
    motion = osim.Storage(motionPath)
    motion.setInDegrees(False)

    # Results  states in motionTrajectory are in SimTK:Stage:Instance
    motionTrajectory = osim.StatesTrajectory.createFromStatesStorage(model, motion, True, True)
    print("Trajectory size: ", motionTrajectory.getSize(), " is compatible: ", motionTrajectory.isCompatibleWith(model))

    poseTrajectory = []
    times = []
    for i in range(motionTrajectory.getSize()):
        motionState = motionTrajectory.get(i)
        model.realizePosition(motionState)
        times.append(motionState.getTime())

        # Loop through bodies in model.
        bodyPoses = []
        for body in model.getBodyList():
            positionGround = body.getPositionInGround(motionState)
            rotationGround = body.getRotationInGround(motionState)
            bodyPoses.append((positionGround, rotationGround))
        poseTrajectory.append(bodyPoses)
    print("Created motion trajectory: ", len(times), "samples.")

    if outputPath != None:
        file = open(outputPath, "w", newline='')
        writer = csv.writer(file)
        writer.writerow(bodyNames)
        for i in range(len(times)):
            row = str(times[i])
            print(f"Time: {row}")
            writer.writerow(row)
        file.close()


    return (bodyNames, times, poseTrajectory)


def main(argv):

    print(f"OpenSim version: {osim.GetVersionAndDate()}")

    sessionPath=""
    modelPath = "./Model/LaiArnoldModified2017_poly_withArms_weldHand_scaled_adjusted.osim"
    motionPath = "./Motions/kinematics_activations_left_leg_squat_0.mot"
    inputPath = sessionPath + motionPath
    outputPath = os.path.splitext(inputPath)[0] + ".csv"
    optionsDict = dict()

    opts, args = getopt.getopt(argv,"im:o:",["input=","model=","output="])
    for opt, arg in opts:
        if opt == "-h":
            print("mot2quats.py -i <inputFile> -o <outputFile> -m <modelFile>")
            sys.exit()
        elif opt in ("-i", "--input"):
            inputPath = arg
        elif opt in ("-o", "--output"):
            outputPath = arg
        elif opt in ("-m", "--model"):
            modelPath = arg

    bodyPoseTrajectories = mot2quats(inputPath, outputPath, modelPath, optionsDict)

# Checks if running this file from a script vs. a module. Useful if planning to use this file also as a module
# to incorporate into other scripts.
if __name__ == "__main__":
    main(sys.argv[1:])