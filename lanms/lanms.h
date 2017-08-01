#pragma once

#include "clipper/clipper.hpp"

// locality-aware NMS
namespace lanms {

	namespace cl = ClipperLib;

	struct Polygon {
		cl::Path poly;
		float score;
	};

	float paths_area(const ClipperLib::Paths &ps) {
		float area = 0;
		for (auto &&p: ps)
			area += cl::Area(p);
		return area;
	}

	float poly_iou(const Polygon &a, const Polygon &b) {
		cl::Clipper clpr;
		clpr.AddPath(a.poly, cl::ptSubject, true);
		clpr.AddPath(b.poly, cl::ptClip, true);

		cl::Paths inter, uni;
		clpr.Execute(cl::ctIntersection, inter, cl::pftEvenOdd);
		clpr.Execute(cl::ctUnion, uni, cl::pftEvenOdd);

		auto inter_area = paths_area(inter),
			 uni_area = paths_area(uni);
		return std::abs(inter_area) / std::max(std::abs(uni_area), 1.0f);
	}

	bool should_merge(const Polygon &a, const Polygon &b, float iou_threshold) {
		return poly_iou(a, b) > iou_threshold;
	}

	class PolyMerger {
		public:
			PolyMerger(): score(0), nr_polys(0) {
				memset(data, 0, sizeof(data));
			}

			void add(const Polygon &p_given) {
				Polygon p;
				if (nr_polys > 0) {
					p = normalize_poly(get(), p_given);
				} else {
					p = p_given;
				}
				assert(p.poly.size() == 4);
				auto &poly = p.poly;
				auto s = p.score;
				data[0] += poly[0].X * s;
				data[1] += poly[0].Y * s;

				data[2] += poly[1].X * s;
				data[3] += poly[1].Y * s;

				data[4] += poly[2].X * s;
				data[5] += poly[2].Y * s;

				data[6] += poly[3].X * s;
				data[7] += poly[3].Y * s;

				score += p.score;

				nr_polys += 1;
			}

			inline std::int64_t sqr(std::int64_t x) { return x * x; }

			Polygon normalize_poly(
					const Polygon &ref,
					const Polygon &p) {

				std::int64_t min_d = std::numeric_limits<std::int64_t>::max();
				size_t best_start = 0, best_order = 0;

				for (size_t start = 0; start < 4; start ++) {
					size_t j = start;
					std::int64_t d = (
							sqr(ref.poly[(j + 0) % 4].X - p.poly[(j + 0) % 4].X)
							+ sqr(ref.poly[(j + 0) % 4].Y - p.poly[(j + 0) % 4].Y)
							+ sqr(ref.poly[(j + 1) % 4].X - p.poly[(j + 1) % 4].X)
							+ sqr(ref.poly[(j + 1) % 4].Y - p.poly[(j + 1) % 4].Y)
							+ sqr(ref.poly[(j + 2) % 4].X - p.poly[(j + 2) % 4].X)
							+ sqr(ref.poly[(j + 2) % 4].Y - p.poly[(j + 2) % 4].Y)
							+ sqr(ref.poly[(j + 3) % 4].X - p.poly[(j + 3) % 4].X)
							+ sqr(ref.poly[(j + 3) % 4].Y - p.poly[(j + 3) % 4].Y)
							);
					if (d < min_d) {
						min_d = d;
						best_start = start;
						best_order = 0;
					}

					d = (
							sqr(ref.poly[(j + 0) % 4].X - p.poly[(j + 3) % 4].X)
							+ sqr(ref.poly[(j + 0) % 4].Y - p.poly[(j + 3) % 4].Y)
							+ sqr(ref.poly[(j + 1) % 4].X - p.poly[(j + 2) % 4].X)
							+ sqr(ref.poly[(j + 1) % 4].Y - p.poly[(j + 2) % 4].Y)
							+ sqr(ref.poly[(j + 2) % 4].X - p.poly[(j + 1) % 4].X)
							+ sqr(ref.poly[(j + 2) % 4].Y - p.poly[(j + 1) % 4].Y)
							+ sqr(ref.poly[(j + 3) % 4].X - p.poly[(j + 0) % 4].X)
							+ sqr(ref.poly[(j + 3) % 4].Y - p.poly[(j + 0) % 4].Y)
						);
					if (d < min_d) {
						min_d = d;
						best_start = start;
						best_order = 1;
					}
				}

				Polygon r;
				r.poly.resize(4);
				auto j = best_start;
				if (best_order == 0) {
					for (size_t i = 0; i < 4; i ++)
						r.poly[i] = p.poly[(j + i) % 4];
				} else {
					for (size_t i = 0; i < 4; i ++)
						r.poly[i] = p.poly[(j + 4 - i - 1) % 4];
				}
				r.score = p.score;
				return r;
			}

			Polygon get() const {
				Polygon p;

				auto &poly = p.poly;
				poly.resize(4);
				auto score_inv = 1.0f / std::max(1e-8f, score);
				poly[0].X = data[0] * score_inv;
				poly[0].Y = data[1] * score_inv;
				poly[1].X = data[2] * score_inv;
				poly[1].Y = data[3] * score_inv;
				poly[2].X = data[4] * score_inv;
				poly[2].Y = data[5] * score_inv;
				poly[3].X = data[6] * score_inv;
				poly[3].Y = data[7] * score_inv;

				assert(score > 0);
				p.score = score;

				return p;
			}

		private:
			std::int64_t data[8];
			float score;
			std::int32_t nr_polys;
	};


	class DisjointSet {
		public:
			DisjointSet(size_t size): m_parent(size) {
				std::iota(std::begin(m_parent), std::end(m_parent), 0);
			}

			bool test(size_t a, size_t b) {
				assert(a < size() && b < size());
				return get_root(a) == get_root(b);
			}

			void merge(size_t a, size_t b) {
				assert(a < size() && b < size());
				m_parent[get_root(a)] = get_root(b);
			}

			size_t get_root(size_t x) {
				assert(x < size());
				return x == m_parent[x] ? x : (m_parent[x] = get_root(m_parent[x]));
			}

			std::vector<std::vector<size_t>> get_groups() {
				std::vector<std::pair<size_t, size_t>> root2id;
				for (size_t i = 0; i < size(); i ++) {
					root2id.emplace_back(std::make_pair(get_root(i), i));
				}
				std::sort(std::begin(root2id), std::end(root2id));

				std::vector<std::vector<size_t>> groups;
				size_t last_root = std::numeric_limits<size_t>::max();
				for (auto &&p: root2id) {
					if (last_root != p.first) {
						groups.emplace_back();
					}
					auto &g = groups.back();
					g.emplace_back(p.second);
					last_root = p.first;
				}
				return groups;
			}

			inline size_t size() const {
				return m_parent.size();
			}


		private:
			std::vector<size_t> m_parent;
	};


	std::vector<Polygon> naive_merge(std::vector<Polygon> &polys, float iou_threshold) {
		std::sort(std::begin(polys), std::end(polys),
				[](const Polygon &a, const Polygon &b) {
					return a.score > b.score;
				});
		auto n = polys.size();
		DisjointSet ds(n);
		for (size_t i = 0; i < n; i ++) {
			for (size_t j = i + 1; j < n; j ++) {
				if (ds.test(i, j))
					continue;
				if (should_merge(polys[i], polys[j], iou_threshold)) {
					ds.merge(i, j);
				}
			}
		}

		auto groups = ds.get_groups();
		std::vector<Polygon> ret;
		for (auto &&g: groups) {
			PolyMerger merger;
			for (auto &&i: g) {
				merger.add(polys[i]);
			}
			ret.emplace_back(merger.get());
		}

		return ret;
 	}

	std::vector<Polygon>
		merge_quadrangle_n9(const float *data, size_t n, float iou_threshold) {
			using cInt = cl::cInt;

			// first pass
			std::vector<Polygon> polys;
			for (size_t i = 0; i < n; i ++) {
				auto p = data + i * 9;
				Polygon poly{
					{
						{cInt(p[0]), cInt(p[1])},
						{cInt(p[2]), cInt(p[3])},
						{cInt(p[4]), cInt(p[5])},
						{cInt(p[6]), cInt(p[7])},
					},
					p[8],
				};

				if (polys.size()) {
					// merge with the last one
					auto &bpoly = polys.back();
					if (should_merge(poly, bpoly, iou_threshold)) {
						PolyMerger merger;
						merger.add(bpoly);
						merger.add(poly);
						bpoly = merger.get();
					} else {
						polys.emplace_back(poly);
					}
				} else {
					polys.emplace_back(poly);
				}
			}
			return naive_merge(polys, iou_threshold);
		}
}
