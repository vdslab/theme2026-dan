// Primal polygon energies, Qu et al. 2021/2022, equations (1)-(6),(9)-(10).
// Original implementation for this repository; analytic piecewise gradients.
// No cardinality or pair-count normalization is applied to the paper terms.
#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <set>
#include <vector>
#include <numeric>

constexpr double PI = 3.14159265358979323846;
struct V {
    double x=0, y=0;
    V operator+(V b) const { return {x+b.x,y+b.y}; }
    V operator-(V b) const { return {x-b.x,y-b.y}; }
    V operator*(double s) const { return {x*s,y*s}; }
    V operator/(double s) const { return {x/s,y/s}; }
    V& operator+=(V b) { x+=b.x; y+=b.y; return *this; }
};
double dot(V a,V b) { return a.x*b.x+a.y*b.y; }
double cross(V a,V b) { return a.x*b.y-a.y*b.x; }
double norm(V a) { return std::sqrt(dot(a,a)+1e-24); }
V cw(V a) { return {a.y,-a.x}; }
V point(const double* x,int i) { return {x[2*i],x[2*i+1]}; }
void add(double* g,int i,V v) { if(g) { g[2*i]+=v.x; g[2*i+1]+=v.y; } }

struct Geo {
    std::vector<int> ids;
    std::vector<int> hull;
    std::vector<double> lengths;
    double area=0, perimeter=0;
    V center, mean;
};
V centroid(const std::vector<int>& ids,const double* x) {
    V numerator, mean;
    double twice=0;
    int n=(int)ids.size();
    for(int i=0;i<n;i++) {
        V a=point(x,ids[i]),b=point(x,ids[(i+1)%n]);
        double c=cross(a,b); twice+=c; numerator+=(a+b)*c; mean+=a;
    }
    return std::abs(twice)>1e-12 ? numerator/(3*twice) : mean/std::max(n,1);
}
Geo geometry(const std::vector<int>& members,const double* x,bool reorder=true) {
    Geo g; g.ids=members;
    int n=(int)members.size();
    if(!n) return g;
    if(n>=3) {
        std::vector<int> order=members, hull;
        std::sort(order.begin(),order.end(),[&](int a,int b) {
            V p=point(x,a),q=point(x,b);
            return p.x!=q.x ? p.x<q.x : (p.y!=q.y ? p.y<q.y : a<b);
        });
        order.erase(std::unique(order.begin(),order.end(),[&](int a,int b){
            V p=point(x,a),q=point(x,b); return p.x==q.x&&p.y==q.y;
        }),order.end());
        for(int a:order) {
            while(hull.size()>1 && cross(point(x,hull.back())-point(x,hull[hull.size()-2]),point(x,a)-point(x,hull.back()))<=0) hull.pop_back();
            hull.push_back(a);
        }
        size_t lower=hull.size();
        for(int k=(int)order.size()-2;k>=0;k--) {
            int a=order[k];
            while(hull.size()>lower && cross(point(x,hull.back())-point(x,hull[hull.size()-2]),point(x,a)-point(x,hull.back()))<=0) hull.pop_back();
            hull.push_back(a);
        }
        if(hull.size()>1) hull.pop_back();
        g.hull=hull;
        V c=centroid(hull,x);
        if(reorder) std::sort(g.ids.begin(),g.ids.end(),[&](int a,int b) {
            V p=point(x,a)-c,q=point(x,b)-c;
            double aa=std::atan2(p.y,p.x),ab=std::atan2(q.y,q.x);
            if(aa!=ab) return aa<ab;
            double ra=dot(p,p),rb=dot(q,q);
            return ra!=rb ? ra<rb : a<b;
        });
    }
    for(int i=0;i<n;i++) {
        V a=point(x,g.ids[i]),b=point(x,g.ids[(i+1)%n]);
        g.area+=cross(a,b)*.5; g.mean+=a;
        g.lengths.push_back(norm(a-b)); g.perimeter+=g.lengths.back();
    }
    g.mean=g.mean/n; g.center=centroid(g.ids,x);
    return g;
}
// Chain rule for the polygon area centroid, not the vertex arithmetic mean.
void centroid_gradient(const Geo& f,const double* x,V v,double* grad) {
    int n=(int)f.ids.size();
    if(!n||!grad) return;
    if(std::abs(f.area)<5e-13) {
        for(int a:f.ids) add(grad,a,v/n);
        return;
    }
    double twice=2*f.area;
    for(int i=0;i<n;i++) {
        int a=f.ids[i],b=f.ids[(i+1)%n]; V p=point(x,a),q=point(x,b);
        double c=cross(p,q),t=dot(v,p+q)/(3*twice)-dot(v,f.center)/twice;
        add(grad,a,v*(c/(3*twice))+cw(q)*t);
        add(grad,b,v*(c/(3*twice))-cw(p)*t);
    }
}
double regularity(const Geo& f,const double* x,double weight,double* grad) {
    int n=(int)f.ids.size(); if(n<3) return 0;
    double c=4*n*std::tan(PI/n);
    if(grad&&weight) for(int i=0;i<n;i++) {
        int j=(i+1)%n,a=f.ids[i],b=f.ids[j]; V p=point(x,a),q=point(x,b);
        V dl=(p-q)*(2*f.perimeter/f.lengths[i]);
        add(grad,a,(dl-cw(q)*(c*.5))*weight);
        add(grad,b,(dl*(-1)+cw(p)*(c*.5))*weight);
    }
    return f.perimeter*f.perimeter-c*f.area;
}
double area_energy(const Geo& f,const double* x,double weight,double* grad) {
    int n=(int)f.ids.size(); if(n<2) return 0; double e=0;
    for(int i=0;i<n;i++) {
        int a=f.ids[i],b=f.ids[(i+1)%n]; double d=f.lengths[i]-1; e+=d*d;
        V force=(point(x,a)-point(x,b))*(2*d/f.lengths[i]*weight);
        add(grad,a,force); add(grad,b,force*(-1));
    }
    return e;
}
// Optional extension: the area missing from the convex hull. Its minimum is
// zero; all original vertices still belong to the actual polygon boundary.
double convexity(const Geo& f,const double* x,double weight,double* grad) {
    if(f.ids.size()<3) return 0;
    double hull_area=0;
    for(size_t i=0;i<f.hull.size();i++) {
        int a=f.hull[i],b=f.hull[(i+1)%f.hull.size()]; V p=point(x,a),q=point(x,b);
        hull_area+=cross(p,q)*.5;
        add(grad,a,cw(q)*(.5*weight));add(grad,b,cw(p)*(-.5*weight));
    }
    for(size_t i=0;i<f.ids.size();i++) {
        int a=f.ids[i],b=f.ids[(i+1)%f.ids.size()];
        add(grad,a,cw(point(x,b))*(-.5*weight));add(grad,b,cw(point(x,a))*(.5*weight));
    }
    return hull_area-f.area;
}
// Optional extension: fit an equal-side regular n-gon using the optimal rigid
// rotation and translation. The envelope theorem gives the residual gradient.
double regular_fit(const Geo& f,const double* x,double weight,double* grad) {
    int n=(int)f.ids.size(); if(n<3) return 0;
    // Sizes are constant for a graph. Avoid millions of redundant trig calls
    // during an exhaustive swap sweep. This kernel is called serially.
    static std::map<int,std::vector<V>> templates;
    auto it=templates.find(n);
    if(it==templates.end()) {
        double r=.5/std::sin(PI/n); std::vector<V> t;
        for(int i=0;i<n;i++) t.push_back({r*std::cos(2*PI*i/n),r*std::sin(2*PI*i/n)});
        it=templates.emplace(n,std::move(t)).first;
    }
    const auto& targets=it->second;
    double a=0,b=0,e=0;
    for(int i=0;i<n;i++) {
        V t=targets[i];
        V p=point(x,f.ids[i])-f.mean; a+=dot(t,p); b+=cross(t,p);
    }
    double angle=std::atan2(b,a),co=std::cos(angle),si=std::sin(angle);
    for(int i=0;i<n;i++) {
        V t=targets[i];
        V rotated={co*t.x-si*t.y,si*t.x+co*t.y};
        V residual=point(x,f.ids[i])-f.mean-rotated;
        e+=dot(residual,residual); add(grad,f.ids[i],residual*(2*weight));
    }
    return e;
}
struct Pair { int a,b,count,pivot,common; };
struct Context {
    int n,m;
    double distance_buffer=.1, angle_buffer=.03;
    std::vector<std::vector<int>> members;
    std::vector<Pair> pairs;
    std::vector<std::vector<int>> node_faces,node_pairs;
    Context(int n_,int m_,const int* offsets,const int* ids): n(n_),m(m_) {
        for(int i=0;i<m;i++) members.emplace_back(ids+offsets[i],ids+offsets[i+1]);
        std::map<std::vector<int>,int> intersections;
        for(int i=0;i<m;i++) for(int j=i+1;j<m;j++) {
            if(members[i].size()<2 || members[j].size()<2) continue;
            std::vector<int> a=members[i],b=members[j],common;
            std::sort(a.begin(),a.end()); std::sort(b.begin(),b.end());
            std::set_intersection(a.begin(),a.end(),b.begin(),b.end(),std::back_inserter(common));
            Pair p={i,j,(int)common.size(),common.empty()?-1:common[0],-1};
            if(p.count>=3) {
                auto it=intersections.find(common);
                if(it==intersections.end()) { p.common=(int)members.size(); intersections[common]=p.common; members.push_back(common); }
                else p.common=it->second;
            }
            pairs.push_back(p);
        }
        node_faces.resize(n); node_pairs.resize(n);
        for(int f=0;f<(int)members.size();f++) for(int a:members[f]) node_faces[a].push_back(f);
        for(int p=0;p<(int)pairs.size();p++) {
            std::set<int> nodes(members[pairs[p].a].begin(),members[pairs[p].a].end());
            nodes.insert(members[pairs[p].b].begin(),members[pairs[p].b].end());
            for(int a:nodes) node_pairs[a].push_back(p);
        }
    }
};

// Post-layout placement extension. Support intervals in fixed directions
// approximate convex-hull penetration; all gradients are piecewise analytic.
// The original polygon shape is retained by a best rigid-fit residual, with
// one common coordinate pair for every original node (no per-face copies).
struct PlacementTarget {
    std::vector<int> ids;
    std::vector<V> reference;
    double scale=1, radius=1;
    double area=0, ratio=1, hull_deficit=0;
};
struct PlacementPair {int a,b,common; double scale;};
struct PlacementContext {
    int n,k;
    double tolerance=.10, mean_ratio=1;
    std::vector<PlacementTarget> targets;
    std::vector<PlacementPair> pairs;
    std::vector<V> axes;
    PlacementContext(int n_,int m,const int* offsets,const int* ids,const double* x,int k_):n(n_),k(k_) {
        for(int i=0;i<k;i++) axes.push_back({std::cos(PI*i/k),std::sin(PI*i/k)});
        for(int f=0;f<m;f++) {
            if(offsets[f+1]-offsets[f]<2) continue;
            PlacementTarget t; V center;
            t.ids.assign(ids+offsets[f],ids+offsets[f+1]);
            for(int id:t.ids) center+=point(x,id);
            center=center/t.ids.size(); t.scale=0;
            for(int id:t.ids) {V p=point(x,id)-center;t.reference.push_back(p);t.scale+=dot(p,p);}
            t.scale=std::max(t.scale,1e-12);t.radius=std::sqrt(t.scale/t.ids.size());
            if(t.ids.size()>=3) {
                Geo g=geometry(t.ids,x,false);t.area=std::max(g.area,1e-12);
                t.ratio=g.perimeter*g.perimeter/(4*t.ids.size()*std::tan(PI/t.ids.size())*t.area);
                double hull_area=0;
                for(size_t z=0;z<g.hull.size();z++) hull_area+=cross(point(x,g.hull[z]),point(x,g.hull[(z+1)%g.hull.size()]))*.5;
                t.hull_deficit=std::max(0.,1-t.area/std::max(hull_area,1e-12));
            }
            targets.push_back(t);
        }
        double total=0;int count=0;
        for(const auto& t:targets) if(t.ids.size()>=3) {total+=t.ratio;count++;}
        mean_ratio=count?total/count:1;
        for(int a=0;a<(int)targets.size();a++) for(int b=a+1;b<(int)targets.size();b++) {
            if(targets[a].ids.size()<3||targets[b].ids.size()<3) continue;
            int count=0;
            for(int id:targets[a].ids) if(std::find(targets[b].ids.begin(),targets[b].ids.end(),id)!=targets[b].ids.end()) count++;
            if(count<=2) {
                double radius=targets[a].radius+targets[b].radius;
                pairs.push_back({a,b,count,std::max(radius*radius,1e-12)});
            }
        }
    }
};
double arc_energy(const Geo& f,const Geo& common,const double* x,double weight,double* grad) {
    std::vector<int> positions; int n=(int)f.ids.size(),k=(int)common.ids.size();
    for(int i=0;i<n;i++) if(std::find(common.ids.begin(),common.ids.end(),f.ids[i])!=common.ids.end()) positions.push_back(i);
    double total=0;
    for(int z=0;z<k;z++) {
        int start=positions[z],end=positions[(z+1)%k],j=start; double length=0;
        do { length+=f.lengths[j]; j=(j+1)%n; } while(j!=end);
        double d=length-double(n)/k; total+=d*d; j=start;
        if(grad&&weight) do {
            int a=f.ids[j],b=f.ids[(j+1)%n];
            V force=(point(x,a)-point(x,b))*(2*d/f.lengths[j]*weight);
            add(grad,a,force); add(grad,b,force*(-1)); j=(j+1)%n;
        } while(j!=end);
    }
    return total;
}

double pair_value(const Context& c,const Pair& pair,const std::vector<Geo>& fs,const double* x,const double* w) {
    const Geo& a=fs[pair.a];const Geo& b=fs[pair.b];
    int na=(int)a.ids.size(),nb=(int)b.ids.size();
    if(pair.count>=3) {
        const Geo& common=fs[pair.common];
        return w[3]*(regularity(common,x,0,nullptr)+arc_energy(a,common,x,0,nullptr)+arc_energy(b,common,x,0,nullptr));
    }
    double required, distance;
    if(pair.count==1) {
        V av=a.center-point(x,pair.pivot),bv=b.center-point(x,pair.pivot);
        distance=std::acos(std::clamp(dot(av,bv)/(norm(av)*norm(bv)),-1+1e-12,1-1e-12));
        required=PI*(double(na-2)/(2*na)+double(nb-2)/(2*nb))+c.angle_buffer;
    } else {
        distance=norm(a.center-b.center);
        required=pair.count==0 ? .5/std::sin(PI/na)+.5/std::sin(PI/nb)+c.distance_buffer : .5/std::tan(PI/na)+.5/std::tan(PI/nb);
    }
    double deficit=std::max(0.,required-distance);return w[2]*deficit*deficit;
}
double shape_value(const Geo& f,const double* x,const double* w) {
    return w[0]*regularity(f,x,0,nullptr)+w[1]*area_energy(f,x,0,nullptr)+(w[4]?w[4]*regular_fit(f,x,0,nullptr):0)
        +(w[6]?w[6]*convexity(f,x,0,nullptr):0);
}

extern "C" {
void* placement_create(int n,int m,const int* offsets,const int* ids,const double* x,int directions) {
    try {return new PlacementContext(n,m,offsets,ids,x,directions);} catch(...) {return nullptr;}
}
void placement_destroy(void* p) {delete static_cast<PlacementContext*>(p);}
void placement_limits(void* p,double tolerance) {static_cast<PlacementContext*>(p)->tolerance=tolerance;}
double placement_evaluate(void* handle,const double* x,double shape_weight,double shared_weight,double margin,
                          double* grad,double* terms) {
    auto& c=*static_cast<PlacementContext*>(handle);
    std::fill(grad,grad+2*c.n,0.);std::fill(terms,terms+4,0.);
    int m=(int)c.targets.size();
    std::vector<double> low(m*c.k),high(m*c.k);
    std::vector<double> low_gradient(m*c.k),high_gradient(m*c.k);
    std::vector<std::vector<double>> low_weights(m),high_weights(m);
    std::vector<Geo> geometries(m);std::vector<double> ratios(m);
    double mean_ratio=0;int face_count=0;
    for(int f=0;f<m;f++) if(c.targets[f].ids.size()>=3) {
        geometries[f]=geometry(c.targets[f].ids,x,false);const auto& g=geometries[f];
        ratios[f]=g.perimeter*g.perimeter/(4*g.ids.size()*std::tan(PI/g.ids.size())*std::max(g.area,1e-12));
        mean_ratio+=ratios[f];face_count++;
    }
    mean_ratio/=std::max(face_count,1);
    double mean_limit=c.mean_ratio*1.015,mean_excess=std::max(0.,mean_ratio/mean_limit-1);
    terms[3]+=face_count*mean_excess*mean_excess;
    const double guard_weight=100000.;
    for(int f=0;f<m;f++) {
        const auto& t=c.targets[f]; int n=(int)t.ids.size(); V mean;
        low_weights[f].resize(n*c.k);high_weights[f].resize(n*c.k);
        for(int id:t.ids) mean+=point(x,id);
        mean=mean/n;double a=0,b=0;
        for(int i=0;i<n;i++) {V p=point(x,t.ids[i])-mean;a+=dot(t.reference[i],p);b+=cross(t.reference[i],p);}
        double angle=std::atan2(b,a),co=std::cos(angle),si=std::sin(angle);
        for(int i=0;i<n;i++) {
            V r=t.reference[i];V rotated={co*r.x-si*r.y,si*r.x+co*r.y};
            V residual=point(x,t.ids[i])-mean-rotated;
            terms[0]+=dot(residual,residual)/t.scale;
            add(grad,t.ids[i],residual*(2*shape_weight/t.scale));
        }
        if(n>=3) {
            const auto& g=geometries[f];double area=std::max(g.area,1e-12);
            double ratio_limit=t.ratio*(1+c.tolerance*.8);
            double excess=std::max(0.,ratios[f]/ratio_limit-1);
            double ratio_gradient=2*excess/ratio_limit+2*mean_excess/mean_limit;
            double area_ratio=area/t.area;
            double area_deficit=std::min(0.,area_ratio-.85),area_excess=std::max(0.,area_ratio-1.15);
            double hull_area=0;
            for(size_t i=0;i<g.hull.size();i++) hull_area+=cross(point(x,g.hull[i]),point(x,g.hull[(i+1)%g.hull.size()]))*.5;
            hull_area=std::max(hull_area,1e-12);
            double deficit=std::max(0.,1-area/hull_area-t.hull_deficit-.015);
            terms[3]+=excess*excess+area_deficit*area_deficit+area_excess*area_excess+deficit*deficit;
            double da=(-ratio_gradient*ratios[f]/area+2*(area_deficit+area_excess)/t.area-2*deficit/hull_area)*guard_weight;
            if(g.area<=1e-12) da=0;
            double dp=ratio_gradient*2*ratios[f]/g.perimeter*guard_weight;
            double dh=2*deficit*area/(hull_area*hull_area)*guard_weight;
            for(int i=0;i<n;i++) {
                int a=g.ids[i],b=g.ids[(i+1)%n];V p=point(x,a),q=point(x,b);
                V force=(p-q)*(dp/g.lengths[i]);
                add(grad,a,force+cw(q)*(.5*da));add(grad,b,force*(-1)-cw(p)*(.5*da));
            }
            for(size_t i=0;i<g.hull.size();i++) {
                int a=g.hull[i],b=g.hull[(i+1)%g.hull.size()];
                add(grad,a,cw(point(x,b))*(.5*dh));add(grad,b,cw(point(x,a))*(-.5*dh));
            }
        }
        for(int axis=0;axis<c.k;axis++) {
            int index=f*c.k+axis;
            low[index]=high[index]=dot(point(x,t.ids[0]),c.axes[axis]);
            for(int id:t.ids) {
                double projection=dot(point(x,id),c.axes[axis]);
                low[index]=std::min(low[index],projection);
                high[index]=std::max(high[index],projection);
            }
            // Outward log-sum-exp supports avoid arbitrary single-vertex
            // gradients on a flat edge. They conservatively contain the hull.
            double tau=std::max(t.radius*.01,1e-9),sl=0,sh=0;
            for(int i=0;i<n;i++) {
                double projection=dot(point(x,t.ids[i]),c.axes[axis]);
                double wl=std::exp((low[index]-projection)/tau),wh=std::exp((projection-high[index])/tau);
                low_weights[f][axis*n+i]=wl;high_weights[f][axis*n+i]=wh;
                sl+=wl;sh+=wh;
            }
            low[index]-=tau*std::log(sl);high[index]+=tau*std::log(sh);
            for(int i=0;i<n;i++) {low_weights[f][axis*n+i]/=sl;high_weights[f][axis*n+i]/=sh;}
        }
    }
    for(const auto& p:c.pairs) {
        double best=-1e100;int ia=0,ib=0;
        for(int axis=0;axis<c.k;axis++) {
            int a=p.a*c.k+axis,b=p.b*c.k+axis;
            double gap=low[b]-high[a];
            if(gap>best) {best=gap;ia=a;ib=b;}
            gap=low[a]-high[b];
            if(gap>best) {best=gap;ia=b;ib=a;}
        }
        double required=p.common?0:margin*std::sqrt(p.scale);
        double depth=std::max(0.,required-best);
        int category=p.common?2:1;double w=p.common?shared_weight:1.;
        terms[category]+=depth*depth/p.scale;
        double force=2*w*depth/p.scale;
        high_gradient[ia]+=force;low_gradient[ib]-=force;
    }
    for(int f=0;f<m;f++) {
        const auto& t=c.targets[f];int n=(int)t.ids.size();
        for(int axis=0;axis<c.k;axis++) {
            int index=f*c.k+axis;
            if(low_gradient[index]==0&&high_gradient[index]==0) continue;
            for(int i=0;i<n;i++) {
                double force=low_gradient[index]*low_weights[f][axis*n+i]+high_gradient[index]*high_weights[f][axis*n+i];
                add(grad,t.ids[i],c.axes[axis]*force);
            }
        }
    }
    return shape_weight*terms[0]+terms[1]+shared_weight*terms[2]+guard_weight*terms[3];
}
void* polygon_create(int n,int m,const int* offsets,const int* ids) {
    try { return new Context(n,m,offsets,ids); } catch(...) { return nullptr; }
}
void polygon_destroy(void* p) { delete static_cast<Context*>(p); }
void polygon_buffers(void* p,double db,double ab) {
    auto& c=*static_cast<Context*>(p); c.distance_buffer=db; c.angle_buffer=ab;
}
void polygon_centers(void* handle,const double* x,double* centers) {
    auto& c=*static_cast<Context*>(handle);
    for(int i=0;i<c.m;i++) {
        V center=geometry(c.members[i],x).center;
        centers[2*i]=center.x;centers[2*i+1]=center.y;
    }
}
void polygon_centers_gradient(void* handle,const double* x,const double* center_gradient,double* grad) {
    auto& c=*static_cast<Context*>(handle);
    for(int i=0;i<c.m;i++) centroid_gradient(geometry(c.members[i],x),x,point(center_gradient,i),grad);
}
// Exhaustive sweep, evaluating only factors affected by the two vertices.
// Optional targets add BOTH parts of the primal-dual coordination energy.
int polygon_swap_sweep(void* handle,double* x,const double* w,const int* candidates,int count,
                      const double* vertex_targets,const double* face_targets,double dd_weight,double* gain) {
    auto& c=*static_cast<Context*>(handle);
    std::vector<Geo> fs;
    for(const auto& m:c.members) fs.push_back(geometry(m,x));
    std::vector<double> shapes(c.m),pairs(c.pairs.size());
    for(int i=0;i<c.m;i++) shapes[i]=shape_value(fs[i],x,w);
    for(int i=0;i<(int)c.pairs.size();i++) pairs[i]=pair_value(c,c.pairs[i],fs,x,w);
    int accepted=0;*gain=0;
    for(int test=0;test<count;test++) {
        int a=candidates[2*test],b=candidates[2*test+1];
        std::vector<int> dirty_faces,dirty_pairs;
        std::set_union(c.node_faces[a].begin(),c.node_faces[a].end(),c.node_faces[b].begin(),c.node_faces[b].end(),std::back_inserter(dirty_faces));
        std::set_union(c.node_pairs[a].begin(),c.node_pairs[a].end(),c.node_pairs[b].begin(),c.node_pairs[b].end(),std::back_inserter(dirty_pairs));
        double delta=0, before_dd=0;
        if(dd_weight) {
            V av=point(x,a)-point(vertex_targets,a),bv=point(x,b)-point(vertex_targets,b);
            before_dd=dot(av,av)+dot(bv,bv);
            for(int f:dirty_faces) if(f<c.m&&!c.members[f].empty()) {V d=fs[f].center-point(face_targets,f);before_dd+=dot(d,d);}
        }
        std::swap(x[2*a],x[2*b]);std::swap(x[2*a+1],x[2*b+1]);
        std::vector<Geo> old; old.reserve(dirty_faces.size());
        std::vector<double> new_shapes,new_pairs;
        for(int f:dirty_faces) {
            old.push_back(std::move(fs[f]));fs[f]=geometry(c.members[f],x);
            if(f<c.m) {double s=shape_value(fs[f],x,w);new_shapes.push_back(s);delta+=s-shapes[f];}
        }
        for(int p:dirty_pairs) {double e=pair_value(c,c.pairs[p],fs,x,w);new_pairs.push_back(e);delta+=e-pairs[p];}
        if(dd_weight) {
            V av=point(x,a)-point(vertex_targets,a),bv=point(x,b)-point(vertex_targets,b);
            double after_dd=dot(av,av)+dot(bv,bv);
            for(int f:dirty_faces) if(f<c.m&&!c.members[f].empty()) {V d=fs[f].center-point(face_targets,f);after_dd+=dot(d,d);}
            delta+=dd_weight*(after_dd-before_dd);
        }
        if(delta < -1e-8) {
            int k=0;for(int f:dirty_faces) if(f<c.m) shapes[f]=new_shapes[k++];
            for(size_t i=0;i<dirty_pairs.size();i++) pairs[dirty_pairs[i]]=new_pairs[i];
            accepted++;*gain-=delta;
        } else {
            std::swap(x[2*a],x[2*b]);std::swap(x[2*a+1],x[2*b+1]);
            for(size_t i=0;i<dirty_faces.size();i++) fs[dirty_faces[i]]=std::move(old[i]);
        }
    }
    return accepted;
}
double polygon_evaluate(void* handle,const double* x,const double* w,double* grad,double* terms) {
    auto& c=*static_cast<Context*>(handle);
    if(grad) std::fill(grad,grad+2*c.n,0.);
    std::fill(terms,terms+7,0.);
    std::vector<Geo> fs; fs.reserve(c.members.size());
    for(const auto& m:c.members) fs.push_back(geometry(m,x));
    std::vector<V> cg(c.m);
    for(int i=0;i<c.m;i++) {
        terms[0]+=regularity(fs[i],x,w[0],grad);
        terms[1]+=area_energy(fs[i],x,w[1],grad);
        if(w[4]) terms[4]+=regular_fit(fs[i],x,w[4],grad);
        if(w[6]) terms[6]+=convexity(fs[i],x,w[6],grad);
    }
    for(const Pair& pair:c.pairs) {
        const Geo& a=fs[pair.a]; const Geo& b=fs[pair.b];
        int na=(int)a.ids.size(),nb=(int)b.ids.size();
        if(pair.count>=3) {
            const Geo& common=fs[pair.common];
            terms[3]+=regularity(common,x,w[3],grad);
            terms[3]+=arc_energy(a,common,x,w[3],grad)+arc_energy(b,common,x,w[3],grad);
        } else if(pair.count==1) {
            V av=a.center-point(x,pair.pivot),bv=b.center-point(x,pair.pivot);
            double an=norm(av),bn=norm(bv),raw=dot(av,bv)/(an*bn);
            double co=std::clamp(raw,-1+1e-12,1-1e-12),angle=std::acos(co);
            double required=PI*(double(na-2)/(2*na)+double(nb-2)/(2*nb))+c.angle_buffer;
            double deficit=std::max(0.,required-angle); terms[2]+=deficit*deficit;
            double factor=std::abs(raw)<1-1e-12 ? w[2]*2*deficit/std::sqrt(1-co*co) : 0;
            V ga=(bv/(an*bn)-av*(co/(an*an)))*factor;
            V gb=(av/(an*bn)-bv*(co/(bn*bn)))*factor;
            cg[pair.a]+=ga; cg[pair.b]+=gb; add(grad,pair.pivot,(ga+gb)*(-1));
        } else {
            double required=pair.count==0 ? .5/std::sin(PI/na)+.5/std::sin(PI/nb)+c.distance_buffer : .5/std::tan(PI/na)+.5/std::tan(PI/nb);
            V delta=a.center-b.center; double distance=norm(delta);
            double deficit=std::max(0.,required-distance); terms[2]+=deficit*deficit;
            V force=delta*(-2*deficit/distance*w[2]); cg[pair.a]+=force; cg[pair.b]+=force*(-1);
        }
    }
    for(int i=0;i<c.m;i++) centroid_gradient(fs[i],x,cg[i],grad);
    if(w[5]) for(int i=0;i<c.n;i++) for(int j=i+1;j<c.n;j++) {
        V delta=point(x,i)-point(x,j); double distance=norm(delta);
        if(distance>=.15) continue;
        double deficit=.15-distance; terms[5]+=deficit*deficit/(.15*.15);
        V force=delta*(-2*deficit/(.15*.15*distance)*w[5]); add(grad,i,force);add(grad,j,force*(-1));
    }
    double total=0; for(int k=0;k<7;k++) total+=w[k]*terms[k];
    return total;
}
}
