#pragma once

//utility.h
#include <chrono>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <stdlib.h>
#include <vector>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <numeric>

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

namespace PaddleOCR {

    struct OCRPredictResult {
        std::vector<std::vector<int>> box;
        std::string text;
        float score = -1.0;
        float cls_score;
        int cls_label = -1;
    };

    struct StructurePredictResult {
        std::vector<float> box;
        std::vector<std::vector<int>> cell_box;
        std::string type;
        std::vector<OCRPredictResult> text_res;
        std::string html;
        float html_score = -1;
        float confidence;
    };

    class Utility {
    public:
        static std::vector<std::string> ReadDict(const std::string& path);

        static void VisualizeBboxes(const cv::Mat& srcimg,
            const std::vector<OCRPredictResult>& ocr_result,
            const std::string& save_path);

        static void VisualizeBboxes(const cv::Mat& srcimg,
            const StructurePredictResult& structure_result,
            const std::string& save_path);

        template <class ForwardIterator>
        inline static size_t argmax(ForwardIterator first, ForwardIterator last) {
            return std::distance(first, std::max_element(first, last));
        }

        static void GetAllFiles(const char* dir_name,
            std::vector<std::string>& all_inputs);

        static cv::Mat GetRotateCropImage(const cv::Mat& srcimage,
            std::vector<std::vector<int>> box);

        static std::vector<int> argsort(const std::vector<float>& array);

        static std::string basename(const std::string& filename);

        static bool PathExists(const std::string& path);

        static void CreateDir(const std::string& path);

        static void print_result(const std::vector<OCRPredictResult>& ocr_result);

        static cv::Mat crop_image(cv::Mat& img, const std::vector<int>& area);
        static cv::Mat crop_image(cv::Mat& img, const std::vector<float>& area);

        static void sorted_boxes(std::vector<OCRPredictResult>& ocr_result);

        static std::vector<int> xyxyxyxy2xyxy(std::vector<std::vector<int>>& box);
        static std::vector<int> xyxyxyxy2xyxy(std::vector<int>& box);

        static float fast_exp(float x);
        static std::vector<float>
            activation_function_softmax(std::vector<float>& src);
        static float iou(std::vector<int>& box1, std::vector<int>& box2);
        static float iou(std::vector<float>& box1, std::vector<float>& box2);

    private:
        static bool comparison_box(const OCRPredictResult& result1,
            const OCRPredictResult& result2) {
            if (result1.box[0][1] < result2.box[0][1]) {
                return true;
            }
            else if (result1.box[0][1] == result2.box[0][1]) {
                return result1.box[0][0] < result2.box[0][0];
            }
            else {
                return false;
            }
        }
    };

} // namespace PaddleOCR

//clipper.h
// use_lines: Enables line clipping. Adds a very minor cost to performance.
#define use_lines

// use_deprecated: Enables temporary support for the obsolete functions
//#define use_deprecated

#include <cstdlib>
#include <cstring>
#include <functional>
#include <list>
#include <ostream>
#include <queue>
#include <set>
#include <stdexcept>
#include <vector>

namespace ClipperLib {

    enum ClipType { ctIntersection, ctUnion, ctDifference, ctXor };
    enum PolyType { ptSubject, ptClip };
    // By far the most widely used winding rules for polygon filling are
    // EvenOdd & NonZero (GDI, GDI+, XLib, OpenGL, Cairo, AGG, Quartz, SVG, Gr32)
    // Others rules include Positive, Negative and ABS_GTR_EQ_TWO (only in OpenGL)
    // see http://glprogramming.com/red/chapter11.html
    enum PolyFillType { pftEvenOdd, pftNonZero, pftPositive, pftNegative };

#ifdef use_int32
    typedef int cInt;
    static cInt const loRange = 0x7FFF;
    static cInt const hiRange = 0x7FFF;
#else
    typedef signed long long cInt;
    static cInt const loRange = 0x3FFFFFFF;
    static cInt const hiRange = 0x3FFFFFFFFFFFFFFFLL;
    typedef signed long long long64; // used by Int128 class
    typedef unsigned long long ulong64;

#endif

    struct IntPoint {
        cInt X;
        cInt Y;
#ifdef use_xyz
        cInt Z;
        IntPoint(cInt x = 0, cInt y = 0, cInt z = 0) : X(x), Y(y), Z(z) {};
#else
        IntPoint(cInt x = 0, cInt y = 0) : X(x), Y(y) {};
#endif

        friend inline bool operator==(const IntPoint& a, const IntPoint& b) {
            return a.X == b.X && a.Y == b.Y;
        }
        friend inline bool operator!=(const IntPoint& a, const IntPoint& b) {
            return a.X != b.X || a.Y != b.Y;
        }
    };
    //------------------------------------------------------------------------------

    typedef std::vector<IntPoint> Path;
    typedef std::vector<Path> Paths;

    inline Path& operator<<(Path& poly, const IntPoint& p) {
        poly.push_back(p);
        return poly;
    }
    inline Paths& operator<<(Paths& polys, const Path& p) {
        polys.push_back(p);
        return polys;
    }

    std::ostream& operator<<(std::ostream& s, const IntPoint& p);
    std::ostream& operator<<(std::ostream& s, const Path& p);
    std::ostream& operator<<(std::ostream& s, const Paths& p);

    struct DoublePoint {
        double X;
        double Y;
        DoublePoint(double x = 0, double y = 0) : X(x), Y(y) {}
        DoublePoint(IntPoint ip) : X((double)ip.X), Y((double)ip.Y) {}
    };
    //------------------------------------------------------------------------------

#ifdef use_xyz
    typedef void (*ZFillCallback)(IntPoint& e1bot, IntPoint& e1top, IntPoint& e2bot,
        IntPoint& e2top, IntPoint& pt);
#endif

    enum InitOptions {
        ioReverseSolution = 1,
        ioStrictlySimple = 2,
        ioPreserveCollinear = 4
    };
    enum JoinType { jtSquare, jtRound, jtMiter };
    enum EndType {
        etClosedPolygon,
        etClosedLine,
        etOpenButt,
        etOpenSquare,
        etOpenRound
    };

    class PolyNode;
    typedef std::vector<PolyNode*> PolyNodes;

    class PolyNode {
    public:
        PolyNode();
        virtual ~PolyNode() {};
        Path Contour;
        PolyNodes Childs;
        PolyNode* Parent;
        PolyNode* GetNext() const;
        bool IsHole() const;
        bool IsOpen() const;
        int ChildCount() const;

    private:
        // PolyNode& operator =(PolyNode& other);
        unsigned Index; // node index in Parent.Childs
        bool m_IsOpen;
        JoinType m_jointype;
        EndType m_endtype;
        PolyNode* GetNextSiblingUp() const;
        void AddChild(PolyNode& child);
        friend class Clipper; // to access Index
        friend class ClipperOffset;
    };

    class PolyTree : public PolyNode {
    public:
        ~PolyTree() { Clear(); };
        PolyNode* GetFirst() const;
        void Clear();
        int Total() const;

    private:
        // PolyTree& operator =(PolyTree& other);
        PolyNodes AllNodes;
        friend class Clipper; // to access AllNodes
    };

    bool Orientation(const Path& poly);
    double Area(const Path& poly);
    int PointInPolygon(const IntPoint& pt, const Path& path);

    void SimplifyPolygon(const Path& in_poly, Paths& out_polys,
        PolyFillType fillType = pftEvenOdd);
    void SimplifyPolygons(const Paths& in_polys, Paths& out_polys,
        PolyFillType fillType = pftEvenOdd);
    void SimplifyPolygons(Paths& polys, PolyFillType fillType = pftEvenOdd);

    void CleanPolygon(const Path& in_poly, Path& out_poly, double distance = 1.415);
    void CleanPolygon(Path& poly, double distance = 1.415);
    void CleanPolygons(const Paths& in_polys, Paths& out_polys,
        double distance = 1.415);
    void CleanPolygons(Paths& polys, double distance = 1.415);

    void MinkowskiSum(const Path& pattern, const Path& path, Paths& solution,
        bool pathIsClosed);
    void MinkowskiSum(const Path& pattern, const Paths& paths, Paths& solution,
        bool pathIsClosed);
    void MinkowskiDiff(const Path& poly1, const Path& poly2, Paths& solution);

    void PolyTreeToPaths(const PolyTree& polytree, Paths& paths);
    void ClosedPathsFromPolyTree(const PolyTree& polytree, Paths& paths);
    void OpenPathsFromPolyTree(PolyTree& polytree, Paths& paths);

    void ReversePath(Path& p);
    void ReversePaths(Paths& p);

    struct IntRect {
        cInt left;
        cInt top;
        cInt right;
        cInt bottom;
    };

    // enums that are used internally ...
    enum EdgeSide { esLeft = 1, esRight = 2 };

    // forward declarations (for stuff used internally) ...
    struct TEdge;
    struct IntersectNode;
    struct LocalMinimum;
    struct OutPt;
    struct OutRec;
    struct Join;

    typedef std::vector<OutRec*> PolyOutList;
    typedef std::vector<TEdge*> EdgeList;
    typedef std::vector<Join*> JoinList;
    typedef std::vector<IntersectNode*> IntersectList;

    //------------------------------------------------------------------------------

    // ClipperBase is the ancestor to the Clipper class. It should not be
    // instantiated directly. This class simply abstracts the conversion of sets of
    // polygon coordinates into edge objects that are stored in a LocalMinima list.
    class ClipperBase {
    public:
        ClipperBase();
        virtual ~ClipperBase();
        virtual bool AddPath(const Path& pg, PolyType PolyTyp, bool Closed);
        bool AddPaths(const Paths& ppg, PolyType PolyTyp, bool Closed);
        virtual void Clear();
        IntRect GetBounds();
        bool PreserveCollinear() { return m_PreserveCollinear; };
        void PreserveCollinear(bool value) { m_PreserveCollinear = value; };

    protected:
        void DisposeLocalMinimaList();
        TEdge* AddBoundsToLML(TEdge* e, bool IsClosed);
        virtual void Reset();
        TEdge* ProcessBound(TEdge* E, bool IsClockwise);
        void InsertScanbeam(const cInt Y);
        bool PopScanbeam(cInt& Y);
        bool LocalMinimaPending();
        bool PopLocalMinima(cInt Y, const LocalMinimum*& locMin);
        OutRec* CreateOutRec();
        void DisposeAllOutRecs();
        void DisposeOutRec(PolyOutList::size_type index);
        void SwapPositionsInAEL(TEdge* edge1, TEdge* edge2);
        void DeleteFromAEL(TEdge* e);
        void UpdateEdgeIntoAEL(TEdge*& e);

        typedef std::vector<LocalMinimum> MinimaList;
        MinimaList::iterator m_CurrentLM;
        MinimaList m_MinimaList;

        bool m_UseFullRange;
        EdgeList m_edges;
        bool m_PreserveCollinear;
        bool m_HasOpenPaths;
        PolyOutList m_PolyOuts;
        TEdge* m_ActiveEdges;

        typedef std::priority_queue<cInt> ScanbeamList;
        ScanbeamList m_Scanbeam;
    };
    //------------------------------------------------------------------------------

    class Clipper : public virtual ClipperBase {
    public:
        Clipper(int initOptions = 0);
        bool Execute(ClipType clipType, Paths& solution,
            PolyFillType fillType = pftEvenOdd);
        bool Execute(ClipType clipType, Paths& solution, PolyFillType subjFillType,
            PolyFillType clipFillType);
        bool Execute(ClipType clipType, PolyTree& polytree,
            PolyFillType fillType = pftEvenOdd);
        bool Execute(ClipType clipType, PolyTree& polytree, PolyFillType subjFillType,
            PolyFillType clipFillType);
        bool ReverseSolution() { return m_ReverseOutput; };
        void ReverseSolution(bool value) { m_ReverseOutput = value; };
        bool StrictlySimple() { return m_StrictSimple; };
        void StrictlySimple(bool value) { m_StrictSimple = value; };
        // set the callback function for z value filling on intersections (otherwise Z
        // is 0)
#ifdef use_xyz
        void ZFillFunction(ZFillCallback zFillFunc);
#endif
    protected:
        virtual bool ExecuteInternal();

    private:
        JoinList m_Joins;
        JoinList m_GhostJoins;
        IntersectList m_IntersectList;
        ClipType m_ClipType;
        typedef std::list<cInt> MaximaList;
        MaximaList m_Maxima;
        TEdge* m_SortedEdges;
        bool m_ExecuteLocked;
        PolyFillType m_ClipFillType;
        PolyFillType m_SubjFillType;
        bool m_ReverseOutput;
        bool m_UsingPolyTree;
        bool m_StrictSimple;
#ifdef use_xyz
        ZFillCallback m_ZFill; // custom callback
#endif
        void SetWindingCount(TEdge& edge);
        bool IsEvenOddFillType(const TEdge& edge) const;
        bool IsEvenOddAltFillType(const TEdge& edge) const;
        void InsertLocalMinimaIntoAEL(const cInt botY);
        void InsertEdgeIntoAEL(TEdge* edge, TEdge* startEdge);
        void AddEdgeToSEL(TEdge* edge);
        bool PopEdgeFromSEL(TEdge*& edge);
        void CopyAELToSEL();
        void DeleteFromSEL(TEdge* e);
        void SwapPositionsInSEL(TEdge* edge1, TEdge* edge2);
        bool IsContributing(const TEdge& edge) const;
        bool IsTopHorz(const cInt XPos);
        void DoMaxima(TEdge* e);
        void ProcessHorizontals();
        void ProcessHorizontal(TEdge* horzEdge);
        void AddLocalMaxPoly(TEdge* e1, TEdge* e2, const IntPoint& pt);
        OutPt* AddLocalMinPoly(TEdge* e1, TEdge* e2, const IntPoint& pt);
        OutRec* GetOutRec(int idx);
        void AppendPolygon(TEdge* e1, TEdge* e2);
        void IntersectEdges(TEdge* e1, TEdge* e2, IntPoint& pt);
        OutPt* AddOutPt(TEdge* e, const IntPoint& pt);
        OutPt* GetLastOutPt(TEdge* e);
        bool ProcessIntersections(const cInt topY);
        void BuildIntersectList(const cInt topY);
        void ProcessIntersectList();
        void ProcessEdgesAtTopOfScanbeam(const cInt topY);
        void BuildResult(Paths& polys);
        void BuildResult2(PolyTree& polytree);
        void SetHoleState(TEdge* e, OutRec* outrec);
        void DisposeIntersectNodes();
        bool FixupIntersectionOrder();
        void FixupOutPolygon(OutRec& outrec);
        void FixupOutPolyline(OutRec& outrec);
        bool IsHole(TEdge* e);
        bool FindOwnerFromSplitRecs(OutRec& outRec, OutRec*& currOrfl);
        void FixHoleLinkage(OutRec& outrec);
        void AddJoin(OutPt* op1, OutPt* op2, const IntPoint offPt);
        void ClearJoins();
        void ClearGhostJoins();
        void AddGhostJoin(OutPt* op, const IntPoint offPt);
        bool JoinPoints(Join* j, OutRec* outRec1, OutRec* outRec2);
        void JoinCommonEdges();
        void DoSimplePolygons();
        void FixupFirstLefts1(OutRec* OldOutRec, OutRec* NewOutRec);
        void FixupFirstLefts2(OutRec* InnerOutRec, OutRec* OuterOutRec);
        void FixupFirstLefts3(OutRec* OldOutRec, OutRec* NewOutRec);
#ifdef use_xyz
        void SetZ(IntPoint& pt, TEdge& e1, TEdge& e2);
#endif
    };
    //------------------------------------------------------------------------------

    class ClipperOffset {
    public:
        ClipperOffset(double miterLimit = 2.0, double roundPrecision = 0.25);
        ~ClipperOffset();
        void AddPath(const Path& path, JoinType joinType, EndType endType);
        void AddPaths(const Paths& paths, JoinType joinType, EndType endType);
        void Execute(Paths& solution, double delta);
        void Execute(PolyTree& solution, double delta);
        void Clear();
        double MiterLimit;
        double ArcTolerance;

    private:
        Paths m_destPolys;
        Path m_srcPoly;
        Path m_destPoly;
        std::vector<DoublePoint> m_normals;
        double m_delta, m_sinA, m_sin, m_cos;
        double m_miterLim, m_StepsPerRad;
        IntPoint m_lowest;
        PolyNode m_polyNodes;

        void FixOrientations();
        void DoOffset(double delta);
        void OffsetPoint(int j, int& k, JoinType jointype);
        void DoSquare(int j, int k);
        void DoMiter(int j, int k, double r);
        void DoRound(int j, int k);
    };
    //------------------------------------------------------------------------------

    class clipperException : public std::exception {
    public:
        clipperException(const char* description) : m_descr(description) {}
        virtual ~clipperException() throw() {}
        virtual const char* what() const throw() { return m_descr.c_str(); }

    private:
        std::string m_descr;
    };
    //------------------------------------------------------------------------------

} // ClipperLib namespace


//preprocess
#include <iostream>
#include <vector>

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

namespace PaddleOCR {

    class Normalize {
    public:
        virtual void Run(cv::Mat* im, const std::vector<float>& mean,
            const std::vector<float>& scale, const bool is_scale = true);
    };

    // RGB -> CHW
    class Permute {
    public:
        virtual void Run(const cv::Mat* im, float* data);
    };

    class PermuteBatch {
    public:
        virtual void Run(const std::vector<cv::Mat> imgs, float* data);
    };

    class ResizeImgType0 {
    public:
        virtual void Run(const cv::Mat& img, cv::Mat& resize_img,
            std::string limit_type, int limit_side_len, float& ratio_h,
            float& ratio_w, bool use_tensorrt);
    };

    class CrnnResizeImg {
    public:
        virtual void Run(const cv::Mat& img, cv::Mat& resize_img, float wh_ratio,
            bool use_tensorrt = false,
            const std::vector<int>& rec_image_shape = { 3, 32, 320 });
    };

    class ClsResizeImg {
    public:
        virtual void Run(const cv::Mat& img, cv::Mat& resize_img,
            bool use_tensorrt = false,
            const std::vector<int>& rec_image_shape = { 3, 48, 192 });
    };

    class TableResizeImg {
    public:
        virtual void Run(const cv::Mat& img, cv::Mat& resize_img,
            const int max_len = 488);
    };

    class TablePadImg {
    public:
        virtual void Run(const cv::Mat& img, cv::Mat& resize_img,
            const int max_len = 488);
    };

    class Resize {
    public:
        virtual void Run(const cv::Mat& img, cv::Mat& resize_img, const int h,
            const int w);
    };

} // namespace PaddleOCR


//postprocess
namespace PaddleOCR {

    class DBPostProcessor {
    public:
        void GetContourArea(const std::vector<std::vector<float>>& box,
            float unclip_ratio, float& distance);

        cv::RotatedRect UnClip(std::vector<std::vector<float>> box,
            const float& unclip_ratio);

        float** Mat2Vec(cv::Mat mat);

        std::vector<std::vector<int>>
            OrderPointsClockwise(std::vector<std::vector<int>> pts);

        std::vector<std::vector<float>> GetMiniBoxes(cv::RotatedRect box,
            float& ssid);

        float BoxScoreFast(std::vector<std::vector<float>> box_array, cv::Mat pred);
        float PolygonScoreAcc(std::vector<cv::Point> contour, cv::Mat pred);

        std::vector<std::vector<std::vector<int>>>
            BoxesFromBitmap(const cv::Mat pred, const cv::Mat bitmap,
                const float& box_thresh, const float& det_db_unclip_ratio,
                const std::string& det_db_score_mode);

        std::vector<std::vector<std::vector<int>>>
            FilterTagDetRes(std::vector<std::vector<std::vector<int>>> boxes,
                float ratio_h, float ratio_w, cv::Mat srcimg);

    private:
        static bool XsortInt(std::vector<int> a, std::vector<int> b);

        static bool XsortFp32(std::vector<float> a, std::vector<float> b);

        std::vector<std::vector<float>> Mat2Vector(cv::Mat mat);

        inline int _max(int a, int b) { return a >= b ? a : b; }

        inline int _min(int a, int b) { return a >= b ? b : a; }

        template <class T> inline T clamp(T x, T min, T max) {
            if (x > max)
                return max;
            if (x < min)
                return min;
            return x;
        }

        inline float clampf(float x, float min, float max) {
            if (x > max)
                return max;
            if (x < min)
                return min;
            return x;
        }
    };

    class TablePostProcessor {
    public:
        void init(std::string label_path, bool merge_no_span_structure = true);
        void Run(std::vector<float>& loc_preds, std::vector<float>& structure_probs,
            std::vector<float>& rec_scores, std::vector<int>& loc_preds_shape,
            std::vector<int>& structure_probs_shape,
            std::vector<std::vector<std::string>>& rec_html_tag_batch,
            std::vector<std::vector<std::vector<int>>>& rec_boxes_batch,
            std::vector<int>& width_list, std::vector<int>& height_list);

    private:
        std::vector<std::string> label_list_;
        std::string end = "eos";
        std::string beg = "sos";
    };

    class PicodetPostProcessor {
    public:
        void init(std::string label_path, const double score_threshold = 0.4,
            const double nms_threshold = 0.5,
            const std::vector<int>& fpn_stride = { 8, 16, 32, 64 });
        void Run(std::vector<StructurePredictResult>& results,
            std::vector<std::vector<float>> outs, std::vector<int> ori_shape,
            std::vector<int> resize_shape, int eg_max);
        std::vector<int> fpn_stride_ = { 8, 16, 32, 64 };

    private:
        StructurePredictResult disPred2Bbox(std::vector<float> bbox_pred, int label,
            float score, int x, int y, int stride,
            std::vector<int> im_shape, int reg_max);
        void nms(std::vector<StructurePredictResult>& input_boxes,
            float nms_threshold);

        std::vector<std::string> label_list_;
        double score_threshold_ = 0.4;
        double nms_threshold_ = 0.5;
        int num_class_ = 5;
    };

} // namespace PaddleOCR