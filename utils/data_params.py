parameters = {
                "BK-L23": {
                          "ovrlp": 0.3, 
                           "dor_thresh": 0.3, 
                           "iou_thresh": 0.6, 
                           "iou_thresh_pseudo": 0.6,
                           "radii": {0: 70, 1: 50, 2: 60, 3: 80, 4: 30}, 
                           "classID2name": {0: "Zebra", 1: "Gazelle", 2: "Waterbuck", 3: "Buffalo", 4: "Other"},
                           "classID2name_HN": {0: "Zebra", 1: "Gazelle", 2: "Wbuck", 3: "Buffalo", 4: "Other"},  
                           "img_format": "jpg"
                           },
                "JE-TL19": {
                           "ovrlp": 0.2, 
                           "dor_thresh": 0.9, 
                           "iou_thresh": 0.1,
                           "iou_thresh_pseudo": 0.3, 
                           "radii": {0: 62, 1: 81, 2: 49}, 
                           "classID2name": {0: "Elephant", 1: "Giraffe", 2: "Zebra"}, 
                           "classID2name_HN": {0: "Elephant", 1: "Giraffe", 2: "Zebra"}, 
                           "img_format": "JPG"
                           },
                "EW-IL22": {
                           "ovrlp": 0.1, 
                           "dor_thresh": 0.4, 
                           "iou_thresh": None, 
                           "iou_thresh_pseudo": 0.5,
                           "radii": {0: 50, 1: 50, 2: 37.5, 3: 50, 4: 50}, 
                           "classID2name": {0: "Brant", 1: "Other", 2: "Gull", 3: "Canada", 4: "Emperor"},
                           "classID2name_HN": {0: "Brant", 1: "Other", 2: "Gull", 3: "Canada", 4: "Emperor"},
                           "img_format": "JPG"
                            }
}