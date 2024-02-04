import { Injectable } from "@nestjs/common";
import { SplitImage } from "../entities/splitImage.entity";
import { EntityManager } from "typeorm";
import { UpdateSplitImageDto } from "../dtos/UpdateSplitImage.dto";


@Injectable()
export class SplitImageRepository {
    // save SplitImage
      async saveSplitImage(splitImage: SplitImage, manager: EntityManager){
            try{
                  return await manager.createQueryBuilder(SplitImage, 'SplitImage').insert().into('SplitImage').values(splitImage).execute();
            }catch(err){
                  throw err;
            }     
      }

      // get SplitImage by imageId
      async getSplitImageByImageId(imageId: string, manager: EntityManager){
            try{
                  return await manager.createQueryBuilder(SplitImage, 'SplitImage').where('image_id = :imageId', {imageId}).getMany();
            }catch(err){
                  throw err;
            }
      }

      // update SplitImage By splitImageId
      async updateSplitImage(SplitImageId:string, splitImageToUpdate: SplitImage, manager: EntityManager){
            try{
                  return await manager.createQueryBuilder(SplitImage, 'SplitImage').update().set(splitImageToUpdate).where('split_image_id = :SplitImageId', {SplitImageId}).execute();
            }catch(err){
                  throw err;
            }
      }

      // delete SplitImage By imageId
      async deleteSplitImageByImageId(imageId:string, manager: EntityManager){
            try{
                  return await manager.createQueryBuilder(SplitImage, 'SplitImage').delete().where('image_id = :imageId', {imageId}).execute();
            }catch(err){
                  throw err;
            }
      }
}