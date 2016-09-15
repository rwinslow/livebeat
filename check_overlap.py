def fix_overlap(intervals):
    """Determine total amount of time from intervals

    :param list int intervals: list of start, end times
    """

    # Make changes to interval list based on condition found
    # Record any unique sets that don't overlap with any other interval
    unique = []
    all_unique = False
    while not all_unique:
        test_l, test_h = intervals.pop()
        for i, interval in enumerate(intervals):
            l, h = interval

            # Overlap low
            if (test_l < l) and (l <= test_h) and (test_h <= h):
                intervals[i] = [test_l, h]
                return intervals
                break

            # Overlap high
            elif (test_h > h) and (l <= test_l) and (test_l <= h):
                intervals[i] = [l, test_h]
                return intervals
                break

            # Overlap fully (outside)
            elif (test_l <= l) and (h <= test_h):
                intervals[i] = [test_l, test_h]
                return intervals
                break

            # Overlap fully (inside)
            elif ((l <= test_l) and (test_l <= h)
                and (l <= test_h) and (test_h <= h)):
                return intervals
                break

            # No overlap condition found
            else:
                pass

        # Save all intervals that overlap no other interval
        if i == len(intervals) - 1 and len(intervals) > 1:
            unique.append([test_l, test_h])

    # Add remaining interval as it is unique
    # unique.append(intervals[0])

    # Sum differences and return value
    return unique

# Test cases
test_interval = [[9001, 9071], [9004, 9074], [9647, 9717], [9769, 9839], [12137, 12207], [17174, 17244], [17177, 17247], [17181, 17251], [24511, 24581], [28185, 28255], [29211, 29281], [29215, 29285], [29234, 29304], [29237, 29307], [29240, 29310], [29244, 29314], [29423, 29493], [29426, 29496], [29429, 29499], [30072, 30142], [30119, 30189]]
print(test_interval)
print(fix_overlap(test_interval))
