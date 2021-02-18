#ifndef LANMS_H
#define LANMS_H

#include "clipper/clipper.hpp"
#include <numeric>
#include <cassert>

// locality-aware NMS
namespace lanms {

	namespace cl = ClipperLib;

	struct Polygon {
		cl::Path poly;
		float score;
	};

	float paths_area(const ClipperLib::Paths &ps);

	float poly_iou(const Polygon &a, const Polygon &b);

	bool should_merge(const Polygon &a, const Polygon &b, float iou_threshold);

	/**
	 * Incrementally merge polygons
	 */
	class PolyMerger {
		public:
			PolyMerger();

			/**
			 * Add a new polygon to be merged.
			 */
			void add(const Polygon &p_given);

			inline std::int64_t sqr(std::int64_t x) { return x * x; }

			Polygon normalize_poly(const Polygon &ref, const Polygon &p);

			Polygon get() const;

		private:
			std::int64_t data[8];
			float score;
			std::int32_t nr_polys;
	};

	/**
	 * The standard NMS algorithm.
	 */
	std::vector<Polygon> standard_nms(std::vector<Polygon> &polys, float iou_threshold);

	std::vector<Polygon> merge_quadrangle_n9(const float *data, size_t n, float iou_threshold=0.3);
}

#endif  // LANMS_H
