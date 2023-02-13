# For OpenSim support
import math, os, sys, getopt, numpy
import opensim as osim
import csv
import numpy as np
import quaternion

def saveOutputCSV(outputPath, times, outputNames, outputData):
    file = open(outputPath, "w", newline='')
    writer = csv.writer(file)
    header = []

    for i in range(len(times)):
        row = []

        if i == 0: # Write header
            header.append("time")
            for label in outputNames:
                header.append(label)
            writer.writerow(header)

        row.append(str(times[i]))
        for data in outputData[i]:
            row.append(str(data))
        writer.writerow(row)

    file.close()
def saveMotionCSV(outputPath, bodyNames, times, poseTrajectories, rotationsOnly):
    file = open(outputPath, "w", newline='')
    writer = csv.writer(file)
    header = []

    lastBodyPoses = []
    for i in range(len(times)):
        row = []  # Row is a list of strings

        if i == 0:
            header.append("time")
        row.append(str(times[i]))
        bodyPoses = poseTrajectories[i]
        for body in range(len(bodyPoses)):
            if i == 0: # Add header info
                lastBodyPoses.append(np.quaternion())
                if rotationsOnly == False:
                    header.append(bodyNames[body] + "_x")
                    header.append(bodyNames[body] + "_y")
                    header.append(bodyNames[body] + "_z")
                header.append(bodyNames[body] + "_qw")
                header.append(bodyNames[body] + "_qx")
                header.append(bodyNames[body] + "_qy")
                header.append(bodyNames[body] + "_qz")

            # Add position
            if rotationsOnly == False:
                row.append(str(bodyPoses[body][0][0]))
                row.append(str(bodyPoses[body][0][1]))
                row.append(str(bodyPoses[body][0][2]))

            # Add rotation as quaternion
            # Check Euclidean distance with previous timestep and select antipodal quaternion that minimizes the distance
            # Better continuity of the orientation trajectory makes learning easier
            lastQuat = lastBodyPoses[body]
            currentQuat = bodyPoses[body][1]
            testA = lastQuat.w
            testW = currentQuat.w
            negCurrentQuat = -currentQuat
            distQ = math.sqrt(pow(currentQuat.w-lastQuat.w, 2.0) + pow(currentQuat.x-lastQuat.x, 2.0) + pow(currentQuat.y-lastQuat.y, 2.0) + pow(currentQuat.z-lastQuat.z, 2.0))
            negDistQ = math.sqrt(pow(negCurrentQuat.w-lastQuat.w, 2.0) + pow(negCurrentQuat.x-lastQuat.x, 2.0) + pow(negCurrentQuat.y-lastQuat.y, 2.0) + pow(negCurrentQuat.z-lastQuat.z, 2.0))
            chosenQuat = currentQuat
            if negDistQ < distQ:
                chosenQuat = negCurrentQuat

            lastBodyPoses[body] = chosenQuat # Store the chosen quaternion this timestep to compare next frame

            # Uncomment to test trajectory for a specific bone
            # if bodyNames[body] == "ulna_r":
            #    print("Body ", bodyNames[body], " last: ", lastQuat, "choice1: ", currentQuat, " choice2: ", negCurrentQuat, " selected: ", chosenQuat)

            row.append(str(chosenQuat.w))
            row.append(str(chosenQuat.x))
            row.append(str(chosenQuat.y))
            row.append(str(chosenQuat.z))

        if i == 0: # Write header before 1st timestamp data
            writer.writerow(header)
        writer.writerow(row)

    file.close()
def mot2quats(motionPath, outputPath, modelPath, optionsDict):

    print(f"Input OpenSim motion path set to: {motionPath}")

    model = osim.Model(modelPath)
    model.initSystem()

    motion = osim.Storage(motionPath)
    motion.setInDegrees(False)
    motionName = os.path.splitext(os.path.basename(motionPath))[0]
    print (f"Motion name: {motionName}")

    # Results  states in motionTrajectory are in SimTK:Stage:Instance
    motionTrajectory = osim.StatesTrajectory.createFromStatesStorage(model, motion, True, True)
    print("Trajectory size: ", motionTrajectory.getSize(), " is compatible: ", motionTrajectory.isCompatibleWith(model))
    motionState = motionTrajectory.get(0)
    model.realizePosition(motionState) # Need to do this to query positions.

    bodyNames = []
    workingBodyList = [] # Store bodies to export
    for body in model.getBodyList():
        # Skip body if position information is invalid.
        testPosition = body.getPositionInGround(motionState)
        if math.isnan(testPosition[0]) == False:
            bodyNames.append(body.getName())
            workingBodyList.append(body)


    poseTrajectories = []
    times = []

    referenceBody = None
    if "relativeTo" in optionsDict:
        for body in workingBodyList:
            if body.getName() == optionsDict["relativeTo"]:
                referenceBody = body
                print("Making poses relative to ", referenceBody.getName())

    for i in range(motionTrajectory.getSize()):
        motionState = motionTrajectory.get(i)
        model.realizePosition(motionState)
        times.append(motionState.getTime())

        invReferencePose = None
        if referenceBody != None:

             referenceRotation = referenceBody.getRotationInGround(motionState).convertRotationToQuaternion()
             referenceQuat = np.quaternion(referenceRotation.get(0), referenceRotation.get(1), referenceRotation.get(2), referenceRotation.get(3))
             invRotation = referenceQuat.conjugate() # Same as quaternion inverse
             invPosition = referenceBody.getPositionInGround(motionState)
             invPosition[0] = -1.0 * invPosition[0]
             invPosition[1] = -1.0 * invPosition[1]
             invPosition[2] = -1.0 * invPosition[2]
             invReferencePose = (invPosition, invRotation)

        # Loop through bodies in model.
        bodyPoses = []
        for body in workingBodyList:
            positionGround = body.getPositionInGround(motionState)
            rotationGround = body.getRotationInGround(motionState).convertRotationToQuaternion()
            # Convert the OpenSim quaternion to the numpy quaternion which has more useful attributes.
            bodyQuat = np.quaternion(rotationGround.get(0), rotationGround.get(1), rotationGround.get(2), rotationGround.get(3))
            if invReferencePose: # Make pose relative to the reference pose
                 positionGround[0] = positionGround[0] + invReferencePose[0][0]
                 positionGround[1] = positionGround[1] + invReferencePose[0][1]
                 positionGround[2] = positionGround[2] + invReferencePose[0][2]
                 bodyQuat = invReferencePose[1] * bodyQuat
            bodyPoses.append((positionGround, bodyQuat))
        poseTrajectories.append(bodyPoses)

    if outputPath != None:
        saveMotionCSV(outputPath + "_motion.csv", bodyNames, times, poseTrajectories, True)

        outputNames = []
        for force in model.getForceSet():
            outputNames.append(force.getName())

        outputData = []

        # Filter out activations only.
        names = model.getStateVariableNames()
        activationLabels = []
        for j in range(names.getSize()):
            if "/activation" in names.get(j):
                activationLabels.append(names.get(j))

        print(f"Found {len(activationLabels)} activation state variables for output.")

        outputNames = []
        for i in range(len(times)):
            sample = []
            state = motionTrajectory.get(i)
            model.realizeVelocity(state)

            for label in activationLabels:
                if i == 0:
                   muscleLabel = label.replace("/activation","")
                   muscleLabel = muscleLabel.replace("/forceset/","")
                   outputNames.append(muscleLabel)
                activation = model.getStateVariableValue(state, label)
                sample.append(str(activation))
            outputData.append(sample)

        saveOutputCSV(outputPath + "_output.csv", times, outputNames, outputData)



    return motionName, (bodyNames, times, poseTrajectories)


def main(argv):

    print(f"OpenSim version: {osim.GetVersionAndDate()}")

    sessionPath=""
    modelPath = "./Model/LaiArnoldModified2017_poly_withArms_weldHand_scaled_adjusted.osim"
    motionPath = "./Motions/kinematics_activations_left_leg_squat_0.mot"
    inputPath = sessionPath + motionPath
    outputPath = os.path.splitext(inputPath)[0]
    optionsDict = dict()

    optionsDict["relativeTo"] = "pelvis"

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

    (name, bodyPoseTrajectories) = mot2quats(inputPath, outputPath, modelPath, optionsDict)

# Checks if running this file from a script vs. a module. Useful if planning to use this file also as a module
# to incorporate into other scripts.
if __name__ == "__main__":
    main(sys.argv[1:])